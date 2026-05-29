import duckdb
import pandas as pd
import chromadb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "readmitiq.duckdb"
CHROMA_PATH = Path(__file__).resolve().parents[1] / "data" / "chroma"


def load_analytics(con) -> dict:
    """Pre-compute all the statistical summaries the LLM will reason over."""
    print("Computing analytics summaries...")

    analytics = {}

    # overall stats
    analytics["overall"] = con.execute("""
        SELECT
            COUNT(*) as total_admissions,
            ROUND(AVG(READMITTED_30D) * 100, 1) as readmission_rate_pct,
            ROUND(AVG(CLM_TOT_CHRG_AMT), 0) as avg_charges,
            ROUND(AVG(CLM_PMT_AMT), 0) as avg_medicare_payment,
            ROUND(AVG(CLM_UTLZTN_DAY_CNT), 1) as avg_los_days,
            COUNT(DISTINCT PRVDR_STATE_CD) as states_covered
        FROM claims_features
    """).df().to_dict(orient="records")[0]

    # by age group
    analytics["by_age"] = con.execute("""
        SELECT
            AGE_GROUP,
            COUNT(*) as admissions,
            ROUND(AVG(READMITTED_30D) * 100, 1) as readmission_rate_pct,
            ROUND(AVG(CLM_TOT_CHRG_AMT), 0) as avg_charges
        FROM claims_features
        WHERE AGE_GROUP IS NOT NULL
        GROUP BY AGE_GROUP
        ORDER BY AGE_GROUP
    """).df().to_dict(orient="records")

    # by diagnosis category
    analytics["by_diagnosis"] = con.execute("""
        SELECT
            DGNS_CATEGORY,
            COUNT(*) as admissions,
            ROUND(AVG(READMITTED_30D) * 100, 1) as readmission_rate_pct,
            ROUND(AVG(CLM_TOT_CHRG_AMT), 0) as avg_charges
        FROM claims_features
        WHERE DGNS_CATEGORY IS NOT NULL
        GROUP BY DGNS_CATEGORY
        ORDER BY readmission_rate_pct DESC
    """).df().to_dict(orient="records")

    # by state (top 15 and bottom 5)
    analytics["by_state"] = con.execute("""
        SELECT
            PRVDR_STATE_CD as state,
            COUNT(*) as admissions,
            ROUND(AVG(READMITTED_30D) * 100, 1) as readmission_rate_pct,
            ROUND(AVG(CLM_TOT_CHRG_AMT), 0) as avg_charges
        FROM claims_features
        WHERE PRVDR_STATE_CD IS NOT NULL
        GROUP BY PRVDR_STATE_CD
        ORDER BY readmission_rate_pct DESC
    """).df().to_dict(orient="records")

    # by admission type
    analytics["by_admission_type"] = con.execute("""
        SELECT
            ADMISSION_TYPE,
            COUNT(*) as admissions,
            ROUND(AVG(READMITTED_30D) * 100, 1) as readmission_rate_pct,
            ROUND(AVG(CLM_TOT_CHRG_AMT), 0) as avg_charges
        FROM claims_features
        GROUP BY ADMISSION_TYPE
        ORDER BY readmission_rate_pct DESC
    """).df().to_dict(orient="records")

    # by discharge status
    analytics["by_discharge"] = con.execute("""
        SELECT
            DISCHARGE_STATUS,
            COUNT(*) as admissions,
            ROUND(AVG(READMITTED_30D) * 100, 1) as readmission_rate_pct
        FROM claims_features
        GROUP BY DISCHARGE_STATUS
        ORDER BY readmission_rate_pct DESC
        LIMIT 10
    """).df().to_dict(orient="records")

    # cost comparison readmitted vs not
    analytics["cost_by_readmission"] = con.execute("""
        SELECT
            CASE WHEN READMITTED_30D = 1 THEN 'Readmitted' ELSE 'Not Readmitted' END as status,
            COUNT(*) as admissions,
            ROUND(AVG(CLM_TOT_CHRG_AMT), 0) as avg_charges,
            ROUND(AVG(CLM_PMT_AMT), 0) as avg_medicare_payment,
            ROUND(AVG(CLM_UTLZTN_DAY_CNT), 1) as avg_los
        FROM claims_features
        GROUP BY READMITTED_30D
    """).df().to_dict(orient="records")

    # by sex
    analytics["by_sex"] = con.execute("""
        SELECT
            SEX,
            COUNT(*) as admissions,
            ROUND(AVG(READMITTED_30D) * 100, 1) as readmission_rate_pct
        FROM claims_features
        WHERE SEX != 'Unknown'
        GROUP BY SEX
    """).df().to_dict(orient="records")

    print(f"  Computed {len(analytics)} analytics summaries")
    return analytics


def build_documents(analytics: dict) -> list[dict]:
    """
    Convert analytics dictionaries into readable text documents.
    These are what gets embedded and retrieved by the LLM.
    """
    print("Building text documents from analytics...")
    docs = []

    # overall summary
    o = analytics["overall"]
    docs.append({
        "id": "overall_summary",
        "text": f"""Overall ReadmitIQ Summary — Key Statistics:
Overall 30-day readmission rate: {o['readmission_rate_pct']}%.
Total hospital admissions analyzed: {o['total_admissions']:,} across {o['states_covered']} US states.
Average total charges per admission: ${o['avg_charges']:,.0f}.
Average Medicare payment per admission: ${o['avg_medicare_payment']:,.0f}.
Average length of stay: {o['avg_los_days']} days.
The overall readmission rate across all patients, conditions, and states is {o['readmission_rate_pct']}%.
Data source: CMS Synthetic Medicare Claims 2025."""
    })

    # age group document
    age_text = "Readmission rates by patient age group:\n"
    for row in analytics["by_age"]:
        age_text += f"- {row['AGE_GROUP']}: {row['readmission_rate_pct']}% readmission rate, {row['admissions']:,} admissions, avg charges ${row['avg_charges']:,.0f}\n"
    docs.append({"id": "by_age", "text": age_text})

    # diagnosis document
    dgns_text = "Readmission rates by primary diagnosis category:\n"
    for row in analytics["by_diagnosis"]:
        dgns_text += f"- {row['DGNS_CATEGORY']}: {row['readmission_rate_pct']}% readmission rate, {row['admissions']:,} admissions, avg charges ${row['avg_charges']:,.0f}\n"
    docs.append({"id": "by_diagnosis", "text": dgns_text})

    # state document
    state_text = "Readmission rates by US state (sorted highest to lowest):\n"
    for row in analytics["by_state"]:
        state_text += f"- {row['state']}: {row['readmission_rate_pct']}% readmission rate, {row['admissions']:,} admissions\n"
    docs.append({"id": "by_state", "text": state_text})

    # admission type document
    adm_text = "Readmission rates by admission type:\n"
    for row in analytics["by_admission_type"]:
        adm_text += f"- {row['ADMISSION_TYPE']}: {row['readmission_rate_pct']}% readmission rate, {row['admissions']:,} admissions, avg charges ${row['avg_charges']:,.0f}\n"
    docs.append({"id": "by_admission_type", "text": adm_text})

    # discharge status document
    disch_text = "Readmission rates by discharge destination (top 10):\n"
    for row in analytics["by_discharge"]:
        disch_text += f"- {row['DISCHARGE_STATUS']}: {row['readmission_rate_pct']}% readmission rate, {row['admissions']:,} admissions\n"
    docs.append({"id": "by_discharge", "text": disch_text})

    # cost comparison document
    cost_text = "Cost and utilization comparison between readmitted and non-readmitted patients:\n"
    for row in analytics["cost_by_readmission"]:
        cost_text += f"- {row['status']}: avg charges ${row['avg_charges']:,.0f}, avg Medicare payment ${row['avg_medicare_payment']:,.0f}, avg LOS {row['avg_los']} days, {row['admissions']:,} admissions\n"
    docs.append({"id": "cost_comparison", "text": cost_text})

    # sex document
    sex_text = "Readmission rates by patient sex:\n"
    for row in analytics["by_sex"]:
        sex_text += f"- {row['SEX']}: {row['readmission_rate_pct']}% readmission rate, {row['admissions']:,} admissions\n"
    docs.append({"id": "by_sex", "text": sex_text})

    print(f"  Built {len(docs)} documents")

    # chart descriptions
    docs.append({
        "id": "chart_descriptions",
        "text": """ReadmitIQ Dashboard Charts and What They Show:

1. '30-Day Readmission Rate by Age Group' bar chart: Shows readmission rates increasing with age. Under 65: 11%, 65-74: 20.8%, 75-84: 26.2%, 85+: 57.9%. Color coded red to green.

2. 'Readmission Rate by Diagnosis Category' horizontal bar chart: Top 10 diagnosis categories ranked by readmission rate. External Causes highest at 69.1%, followed by Genitourinary 59.3%, Mental Health 57.6%.

3. 'Readmission Rate by Length of Stay' bar chart: Shows how LOS correlates with readmission risk across 5 buckets from 1 day to 15+ days.

4. 'Readmission Rate by Admission Type' bar chart: Compares Emergency, Elective, Urgent, Trauma and Unknown admission types by readmission rate.

5. '30-Day Readmission Rate by State' choropleth map: US map colored by state readmission rate. Darker red = higher readmission rate.

6. 'Avg Charges vs Medicare Payment by Diagnosis' grouped bar chart: Compares what hospitals bill vs what Medicare actually pays across diagnosis categories.

7. 'Avg Cost: Readmitted vs Not Readmitted' bar chart: Readmitted patients avg $2,736 vs non-readmitted $8,416 — counterintuitive finding suggesting sicker patients have shorter initial stays.

8. Model Performance section: Shows baseline logistic regression AUC 0.772 vs XGBoost AUC 0.908, CV mean 0.907 ± 0.004. Top predictive features: Medicare payment amount, total charges, length of stay, age group."""
    })
    return docs


def store_in_chroma(docs: list[dict]):
    """Embed and store documents in ChromaDB vector store."""
    print(f"Storing in ChromaDB at {CHROMA_PATH}...")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # delete existing collection if rebuilding
    try:
        client.delete_collection("readmitiq_analytics")
    except Exception:
        pass

    collection = client.create_collection(
        name="readmitiq_analytics",
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[doc["id"] for doc in docs],
        documents=[doc["text"] for doc in docs],
    )

    print(f"  Stored {collection.count()} documents in ChromaDB")
    return collection


def run():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    analytics = load_analytics(con)
    con.close()

    docs = build_documents(analytics)
    store_in_chroma(docs)
    print("\nEmbeddings complete. ChromaDB ready.")


if __name__ == "__main__":
    run()