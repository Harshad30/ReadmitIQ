import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "readmitiq.duckdb"

# ── ICD-10 Chapter mapping (first character of diagnosis code) ─────────────
# This groups the 70,000+ ICD codes into 21 readable categories
ICD_CHAPTERS = {
    "A": "Infectious Disease", "B": "Infectious Disease",
    "C": "Cancer", "D": "Blood Disorders",
    "E": "Endocrine/Metabolic", "F": "Mental Health",
    "G": "Nervous System", "H": "Eye/Ear",
    "I": "Circulatory", "J": "Respiratory",
    "K": "Digestive", "L": "Skin",
    "M": "Musculoskeletal", "N": "Genitourinary",
    "O": "Pregnancy", "P": "Perinatal",
    "Q": "Congenital", "R": "Symptoms/Signs",
    "S": "Injury/Trauma", "T": "Injury/Trauma",
    "V": "External Causes", "W": "External Causes",
    "X": "External Causes", "Y": "External Causes",
    "Z": "Health Status",
}

# ── Admission type mapping ─────────────────────────────────────────────────
ADMISSION_TYPES = {
    "1": "Emergency",
    "2": "Urgent",
    "3": "Elective",
    "4": "Newborn",
    "5": "Trauma",
    "9": "Unknown",
}

# ── Discharge status mapping ───────────────────────────────────────────────
DISCHARGE_STATUS = {
    "01": "Home",
    "02": "Short Term Hospital",
    "03": "SNF",
    "04": "ICF",
    "05": "Other Facility",
    "06": "Home Health",
    "07": "AMA",
    "20": "Expired",
    "30": "Still Patient",
    "43": "Federal Hospital",
    "50": "Hospice Home",
    "51": "Hospice Medical",
    "61": "Swing Bed",
    "62": "Rehab Facility",
    "63": "Long Term Care",
    "65": "Psychiatric Hospital",
    "66": "Critical Access",
    "69": "Disaster Alternative",
    "81": "VA Facility",
    "82": "State Hospital",
}


def load_tables(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    inpatient = con.execute("SELECT * FROM inpatient_claims").df()
    beneficiary = con.execute("SELECT * FROM beneficiary").df()
    print(f"  Loaded {len(inpatient):,} claims and {len(beneficiary):,} beneficiaries")
    return inpatient, beneficiary


def engineer_features(inpatient: pd.DataFrame, beneficiary: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")

    # ── Join beneficiary demographics ──────────────────────────────────────
    df = inpatient.merge(beneficiary, on="BENE_ID", how="left")
    print(f"  After join: {len(df):,} rows")

    # ── Age groups ─────────────────────────────────────────────────────────
    # AGE_AT_END_REF_YR already calculated by CMS - we just bin it
    df["AGE_GROUP"] = pd.cut(
        df["AGE_AT_END_REF_YR"],
        bins=[0, 64, 74, 84, 200],
        labels=["Under 65", "65-74", "75-84", "85+"]
    )

    # ── Diagnosis category ─────────────────────────────────────────────────
    # map first character of ICD code to readable chapter
    df["DGNS_CATEGORY"] = (
        df["PRNCPAL_DGNS_CD"]
        .astype(str)
        .str[0]
        .str.upper()
        .map(ICD_CHAPTERS)
        .fillna("Other")
    )

    # ── Admission type label ───────────────────────────────────────────────
    df["ADMISSION_TYPE"] = (
        df["CLM_IP_ADMSN_TYPE_CD"]
        .astype(str)
        .map(ADMISSION_TYPES)
        .fillna("Unknown")
    )

    # ── Discharge status label ─────────────────────────────────────────────
    df["DISCHARGE_STATUS"] = (
        df["PTNT_DSCHRG_STUS_CD"]
        .astype(str)
        .str.zfill(2)          # pad to 2 digits e.g. "1" → "01"
        .map(DISCHARGE_STATUS)
        .fillna("Other")
    )

    # ── Cost features ──────────────────────────────────────────────────────
    # how much did medicare actually cover vs what was billed
    df["COST_COVERAGE_RATIO"] = (
        df["CLM_PMT_AMT"] / df["CLM_TOT_CHRG_AMT"]
    ).round(4)

    # ── Length of stay buckets ─────────────────────────────────────────────
    df["LOS_GROUP"] = pd.cut(
        df["CLM_UTLZTN_DAY_CNT"],
        bins=[-1, 1, 3, 7, 14, 1000],
        labels=["1 day", "2-3 days", "4-7 days", "8-14 days", "15+ days"]
    )

    # ── Sex label ──────────────────────────────────────────────────────────
    df["SEX"] = df["SEX_IDENT_CD"].map({1: "Male", 2: "Female"}).fillna("Unknown")

    # ── Race label ─────────────────────────────────────────────────────────
    df["RACE"] = df["BENE_RACE_CD"].map({
        1: "White", 2: "Black", 3: "Other",
        4: "Asian", 5: "Hispanic", 6: "Native American"
    }).fillna("Unknown")

    print(f"  Readmission rate by age group:")
    print(df.groupby("AGE_GROUP", observed=True)["READMITTED_30D"].mean().mul(100).round(1))
    print(f"\n  Readmission rate by diagnosis category (top 5):")
    print(df.groupby("DGNS_CATEGORY")["READMITTED_30D"].mean().mul(100).round(1).nlargest(5))

    return df


def store_features(con, df: pd.DataFrame):
    print("\nStoring enriched feature table...")
    con.execute("DROP TABLE IF EXISTS claims_features")
    con.execute("CREATE TABLE claims_features AS SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM claims_features").fetchone()[0]
    print(f"  Stored {count:,} rows in claims_features table")


def run():
    con = duckdb.connect(str(DB_PATH))
    inpatient, beneficiary = load_tables(con)
    df = engineer_features(inpatient, beneficiary)
    store_features(con, df)
    con.close()
    print("\nFeature engineering complete.")


if __name__ == "__main__":
    run()