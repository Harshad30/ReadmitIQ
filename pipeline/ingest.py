import duckdb
import pandas as pd
from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DB_PATH  = Path(__file__).resolve().parents[1] / "data" / "readmitiq.duckdb"

INPATIENT_FILE   = RAW_DIR / "inpatient.csv"
BENEFICIARY_FILE = RAW_DIR / "beneficiary_2025.csv"

# ── Columns we actually need (drop the noise) ──────────────────────────────
INPATIENT_COLS = [
    "BENE_ID",           # patient identifier
    "CLM_ID",            # unique claim identifier
    "CLM_ADMSN_DT",      # admission date
    "NCH_BENE_DSCHRG_DT", # discharge date
    "PTNT_DSCHRG_STUS_CD", # discharge status (home, transferred, expired etc)
    "PRVDR_STATE_CD",    # state where hospital is
    "ORG_NPI_NUM",       # hospital identifier
    "CLM_DRG_CD",        # DRG code - how hospitals get paid
    "PRNCPAL_DGNS_CD",   # primary diagnosis ICD code
    "CLM_TOT_CHRG_AMT",  # total charges billed
    "CLM_PMT_AMT",       # what medicare actually paid
    "CLM_UTLZTN_DAY_CNT", # length of stay in days
    "CLM_IP_ADMSN_TYPE_CD", # admission type (emergency, elective etc)
]

BENEFICIARY_COLS = [
    "BENE_ID",           # patient identifier - joins to inpatient
    "BENE_BIRTH_DT",     # date of birth
    "SEX_IDENT_CD",      # sex (was BENE_SEX_IDENT_CD)
    "BENE_RACE_CD",      # race
    "STATE_CODE",        # state (was BENE_STATE_CD)
    "COUNTY_CD",         # county (was BENE_COUNTY_CD)
    "AGE_AT_END_REF_YR", # age - already calculated for us, bonus!
    "BENE_DEATH_DT",     # death date - useful for outcome analysis
]


def load_inpatient() -> pd.DataFrame:
    print("Loading inpatient claims...")
    df = pd.read_csv(
        INPATIENT_FILE,
        sep="|",                  # TSV with pipe separator
        usecols=INPATIENT_COLS,   # only load columns we need
        parse_dates=["CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT"],
        dtype={"PRNCPAL_DGNS_CD": str, "CLM_DRG_CD": str},
        low_memory=False
    )
    print(f"  Loaded {len(df):,} inpatient claims")
    return df


def load_beneficiary() -> pd.DataFrame:
    print("Loading beneficiary data...")
    df = pd.read_csv(
        BENEFICIARY_FILE,
        sep="|",
        usecols=BENEFICIARY_COLS,
        parse_dates=["BENE_BIRTH_DT"],
        low_memory=False
    )
    print(f"  Loaded {len(df):,} beneficiaries")
    return df


def clean_inpatient(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning inpatient claims...")

    # drop rows missing critical dates
    before = len(df)
    df = df.dropna(subset=["CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT", "BENE_ID"])
    print(f"  Dropped {before - len(df):,} rows missing critical dates")

    # KEY FIX: deduplicate to one row per claim
    # each claim has multiple billing lines but same admission/discharge dates
    before = len(df)
    df = df.drop_duplicates(subset=["CLM_ID"], keep="first")
    print(f"  Deduplicated {before - len(df):,} billing lines → one row per claim")
    print(f"  Unique claims: {len(df):,}")

    # sanity checks
    df = df[df["CLM_UTLZTN_DAY_CNT"] >= 0]
    df = df[df["CLM_TOT_CHRG_AMT"] > 0]

    # sort by patient and admission date - critical for readmission logic
    df = df.sort_values(["BENE_ID", "CLM_ADMSN_DT"]).reset_index(drop=True)

    print(f"  Clean inpatient shape: {df.shape}")
    return df


def engineer_readmission_flag(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering 30-day readmission flag...")

    df["NEXT_ADMSN_DT"] = df.groupby("BENE_ID")["CLM_ADMSN_DT"].shift(-1)

    df["DAYS_TO_READMISSION"] = (
        df["NEXT_ADMSN_DT"] - df["NCH_BENE_DSCHRG_DT"]
    ).dt.days

    # exclude same-day and negative (transfers/overlapping stays)
    # true readmission = discharged then came back 1-30 days later
    df["READMITTED_30D"] = (
        df["DAYS_TO_READMISSION"].notna() &
        (df["DAYS_TO_READMISSION"] >= 1) &
        (df["DAYS_TO_READMISSION"] <= 30)
    ).astype(int)

    rate = df["READMITTED_30D"].mean() * 100
    print(f"  Overall 30-day readmission rate: {rate:.1f}%")
    return df


def store_in_duckdb(inpatient: pd.DataFrame, beneficiary: pd.DataFrame):
    print(f"Storing in DuckDB at {DB_PATH}...")
    con = duckdb.connect(str(DB_PATH))

    con.execute("DROP TABLE IF EXISTS inpatient_claims")
    con.execute("DROP TABLE IF EXISTS beneficiary")

    con.execute("CREATE TABLE inpatient_claims AS SELECT * FROM inpatient")
    con.execute("CREATE TABLE beneficiary AS SELECT * FROM beneficiary")

    count = con.execute("SELECT COUNT(*) FROM inpatient_claims").fetchone()[0]
    print(f"  Stored {count:,} rows in inpatient_claims table")
    con.close()


def run():
    inpatient   = load_inpatient()
    beneficiary = load_beneficiary()
    inpatient   = clean_inpatient(inpatient)
    inpatient   = engineer_readmission_flag(inpatient)
    store_in_duckdb(inpatient, beneficiary)
    print("\nIngest complete. Database ready at data/readmitiq.duckdb")


if __name__ == "__main__":
    run()