import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat.query import ask, get_collection

# ── Config ─────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "readmitiq.duckdb"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

st.set_page_config(
    page_title="ReadmitIQ",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
def chart_help(text: str):
    with st.popover("ℹ️"):
        st.markdown(text)


# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0066cc;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM claims_features").df()
    con.close()
    return df


@st.cache_data
def load_metrics():
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_importance():
    fi_path = MODELS_DIR / "feature_importance.csv"
    if fi_path.exists():
        return pd.read_csv(fi_path)
    return pd.DataFrame()


# ── Sidebar filters ─────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.sidebar.title("ReadmitIQ")
    st.sidebar.caption("Hospital Readmission Analytics")
    st.sidebar.divider()

    st.sidebar.markdown("### Filters")

    states = ["All"] + sorted(df["PRVDR_STATE_CD"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("State", states)

    age_groups = ["All"] + df["AGE_GROUP"].cat.categories.tolist()
    selected_age = st.sidebar.selectbox("Age Group", age_groups)

    admission_types = ["All"] + sorted(df["ADMISSION_TYPE"].dropna().unique().tolist())
    selected_admission = st.sidebar.selectbox("Admission Type", admission_types)

    dgns_cats = ["All"] + sorted(df["DGNS_CATEGORY"].dropna().unique().tolist())
    selected_dgns = st.sidebar.selectbox("Diagnosis Category", dgns_cats)

    st.sidebar.divider()
    st.sidebar.caption("Data: CMS Synthetic Medicare Claims 2025")

    return selected_state, selected_age, selected_admission, selected_dgns


def apply_filters(df, state, age, admission, dgns):
    if state != "All":
        df = df[df["PRVDR_STATE_CD"] == state]
    if age != "All":
        df = df[df["AGE_GROUP"] == age]
    if admission != "All":
        df = df[df["ADMISSION_TYPE"] == admission]
    if dgns != "All":
        df = df[df["DGNS_CATEGORY"] == dgns]
    return df


# ── KPI cards ───────────────────────────────────────────────────────────────
def render_kpis(df):
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    total = len(df)
    readmitted = df["READMITTED_30D"].sum()
    readmit_rate = df["READMITTED_30D"].mean() * 100
    avg_los = df["CLM_UTLZTN_DAY_CNT"].mean()
    avg_cost = df["CLM_TOT_CHRG_AMT"].mean()
    avg_payment = df["CLM_PMT_AMT"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Admissions</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        color = "#cc0000" if readmit_rate > 25 else "#0066cc"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color}">
            <div class="metric-value" style="color:{color}">{readmit_rate:.1f}%</div>
            <div class="metric-label">30-Day Readmission Rate</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{readmitted:,}</div>
            <div class="metric-label">Readmissions</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_los:.1f}</div>
            <div class="metric-label">Avg LOS — days (synthetic data)</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_cost:,.0f}</div>
            <div class="metric-label">Avg Total Charges</div>
        </div>""", unsafe_allow_html=True)


# ── Charts ───────────────────────────────────────────────────────────────────
def render_charts(df):
    st.divider()
    st.markdown('<div class="section-header">Readmission Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        chart_help("""
        **What it shows:** The 30-day readmission rate for each patient age group.
        
        **How to read it:** Taller bar = more patients from that age group are returning 
        to the hospital within 30 days of discharge. Red bars indicate higher risk.
        
        **Why it matters:** Older patients have more complex conditions and weaker 
        support systems, making them more likely to be readmitted. Hospitals use this 
        to prioritize discharge planning resources.
        """)
        age_data = (
            df.groupby("AGE_GROUP", observed=True)["READMITTED_30D"]
            .agg(["mean", "count"])
            .reset_index()
        )
        age_data["rate"] = age_data["mean"] * 100
        fig = px.bar(
            age_data, x="AGE_GROUP", y="rate",
            title="30-Day Readmission Rate by Age Group",
            labels={"AGE_GROUP": "Age Group", "rate": "Readmission Rate (%)"},
            color="rate",
            color_continuous_scale="RdYlGn_r",
            text=age_data["rate"].apply(lambda x: f"{x:.1f}%")
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        chart_help("""
        **What it shows:** The top 10 diagnosis categories ranked by readmission rate.
        
        **How to read it:** Longer bar = patients with that type of condition are more 
        likely to be readmitted within 30 days. Categories are sorted highest to lowest.
        
        **Why it matters:** Helps hospitals identify which clinical departments need 
        stronger post-discharge protocols and care coordination.
        """)
        dgns_data = (
            df.groupby("DGNS_CATEGORY")["READMITTED_30D"]
            .agg(["mean", "count"])
            .reset_index()
        )
        dgns_data["rate"] = dgns_data["mean"] * 100
        dgns_data = dgns_data.sort_values("rate", ascending=True).tail(10)
        fig = px.bar(
            dgns_data, x="rate", y="DGNS_CATEGORY",
            orientation="h",
            title="Readmission Rate by Diagnosis Category (Top 10)",
            labels={"DGNS_CATEGORY": "", "rate": "Readmission Rate (%)"},
            color="rate",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        chart_help("""
        **What it shows:** How length of hospital stay relates to readmission risk.
        
        **How to read it:** Each bar represents patients grouped by how long they stayed. 
        The height shows what percentage of that group came back within 30 days.
        
        **Why it matters:** Very short stays (1 day) may indicate premature discharge — 
        patients sent home before they're clinically stable, leading to higher readmissions.
        """)
        los_data = (
            df.groupby("LOS_GROUP", observed=True)["READMITTED_30D"]
            .agg(["mean", "count"])
            .reset_index()
        )
        los_data["rate"] = los_data["mean"] * 100
        fig = px.bar(
            los_data, x="LOS_GROUP", y="rate",
            title="Readmission Rate by Length of Stay",
            labels={"LOS_GROUP": "Length of Stay", "rate": "Readmission Rate (%)"},
            color="rate",
            color_continuous_scale="RdYlGn_r",
            text=los_data["rate"].apply(lambda x: f"{x:.1f}%")
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with col4:
        chart_help("""
        **What it shows:** Readmission rates broken down by how the patient was admitted.
        
        **How to read it:** Emergency admissions are unplanned and typically involve 
        sicker patients, so higher readmission rates are expected. Elective admissions 
        are planned procedures with better preparation and lower risk.
        
        **Why it matters:** A hospital with unusually high elective readmissions may 
        have gaps in pre-surgical preparation or post-operative care protocols.
        """)
        adm_data = (
            df.groupby("ADMISSION_TYPE")["READMITTED_30D"]
            .agg(["mean", "count"])
            .reset_index()
        )
        adm_data["rate"] = adm_data["mean"] * 100
        fig = px.bar(
            adm_data, x="ADMISSION_TYPE", y="rate",
            title="Readmission Rate by Admission Type",
            labels={"ADMISSION_TYPE": "Admission Type", "rate": "Readmission Rate (%)"},
            color="rate",
            color_continuous_scale="RdYlGn_r",
            text=adm_data["rate"].apply(lambda x: f"{x:.1f}%")
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")


# ── State map ────────────────────────────────────────────────────────────────
def render_state_map(df):
    st.divider()
    st.markdown('<div class="section-header">Geographic Analysis</div>', unsafe_allow_html=True)

    chart_help("""
    **What it shows:** 30-day readmission rates across all US states.
    
    **How to read it:** Darker red = higher readmission rate in that state. 
    Hover over any state to see the exact rate and number of admissions.
    
    **Why it matters:** Geographic variation in readmission rates can indicate 
    differences in healthcare quality, patient demographics, or access to 
    post-discharge follow-up care across regions.
    """)

    state_data = (
        df.groupby("PRVDR_STATE_CD")["READMITTED_30D"]
        .agg(["mean", "count"])
        .reset_index()
    )
    state_data["rate"] = state_data["mean"] * 100
    state_data.columns = ["state", "mean", "count", "rate"]

    fig = px.choropleth(
        state_data,
        locations="state",
        locationmode="USA-states",
        color="rate",
        scope="usa",
        title="30-Day Readmission Rate by State",
        color_continuous_scale="RdYlGn_r",
        labels={"rate": "Readmission Rate (%)"},
        hover_data={"count": True, "rate": ":.1f"}
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")


# ── Model performance ────────────────────────────────────────────────────────
def render_model_performance(metrics, feature_importance):
    st.divider()
    st.markdown('<div class="section-header">ML Model Performance</div>', unsafe_allow_html=True)

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Baseline AUC (Logistic Reg)", f"{metrics.get('baseline_auc', 0):.3f}")
        with c2:
            st.metric("XGBoost AUC", f"{metrics.get('xgboost_auc', 0):.3f}",
                     delta=f"+{metrics.get('improvement_over_baseline', 0):.3f} vs baseline")
        with c3:
            st.metric("CV Mean AUC", f"{metrics.get('cv_mean_auc', 0):.3f}")
        with c4:
            st.metric("CV Std Dev", f"±{metrics.get('cv_std', 0):.3f}")

    if not feature_importance.empty:
        st.markdown("#### Feature Importance")
        fig = px.bar(
            feature_importance.head(10),
            x="importance", y="feature",
            orientation="h",
            title="Top 10 Predictive Features",
            labels={"importance": "Importance Score", "feature": ""},
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=400
        )
        st.plotly_chart(fig, width="stretch")


# ── Cost analysis ────────────────────────────────────────────────────────────
def render_cost_analysis(df):
    st.divider()
    st.markdown('<div class="section-header">Cost Analysis</div>', unsafe_allow_html=True)
    chart_help("""
    **What it shows:** How hospital costs differ across diagnosis categories 
    and between readmitted vs non-readmitted patients.
    
    **How to read it:** Left chart compares what hospitals bill (total charges) 
    vs what Medicare actually pays across diagnosis types. Right chart compares 
    average costs between patients who were readmitted vs those who weren't.
    
    **Why it matters:** The counterintuitive finding — readmitted patients have 
    LOWER initial charges — suggests they were discharged too quickly before 
    being fully stabilized, leading to costly return visits.
    """)
    col1, col2 = st.columns(2)

    with col1:
        cost_data = (
            df.groupby("DGNS_CATEGORY")[["CLM_TOT_CHRG_AMT", "CLM_PMT_AMT"]]
            .mean()
            .reset_index()
            .sort_values("CLM_TOT_CHRG_AMT", ascending=False)
            .head(10)
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Avg Total Charges",
            x=cost_data["DGNS_CATEGORY"],
            y=cost_data["CLM_TOT_CHRG_AMT"],
            marker_color="#0066cc"
        ))
        fig.add_trace(go.Bar(
            name="Avg Medicare Payment",
            x=cost_data["DGNS_CATEGORY"],
            y=cost_data["CLM_PMT_AMT"],
            marker_color="#00aa44"
        ))
        fig.update_layout(
            title="Avg Charges vs Medicare Payment by Diagnosis",
            barmode="group",
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        readmit_cost = (
            df.groupby("READMITTED_30D")[["CLM_TOT_CHRG_AMT", "CLM_UTLZTN_DAY_CNT"]]
            .mean()
            .reset_index()
        )
        readmit_cost["READMITTED_30D"] = readmit_cost["READMITTED_30D"].map({
            0: "Not Readmitted", 1: "Readmitted"
        })

        fig = px.bar(
            readmit_cost,
            x="READMITTED_30D", y="CLM_TOT_CHRG_AMT",
            title="Avg Cost: Readmitted vs Not Readmitted",
            labels={"READMITTED_30D": "", "CLM_TOT_CHRG_AMT": "Avg Total Charges ($)"},
            color="READMITTED_30D",
            color_discrete_sequence=["#00aa44", "#cc0000"],
            text=readmit_cost["CLM_TOT_CHRG_AMT"].apply(lambda x: f"${x:,.0f}")
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width="stretch")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    metrics = load_metrics()
    feature_importance = load_feature_importance()

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🎯 Risk Scorer","🤖 AI Assistant"],
        index=0
    )

    if page == "📊 Dashboard":
        selected_state, selected_age, selected_admission, selected_dgns = render_sidebar(df)
        filtered_df = apply_filters(df, selected_state, selected_age, selected_admission, selected_dgns)

        st.title("🏥 ReadmitIQ — Hospital Readmission Analytics")
        st.caption(f"Showing {len(filtered_df):,} of {len(df):,} admissions | CMS Synthetic Medicare Claims 2025")

        render_kpis(filtered_df)
        render_charts(filtered_df)
        render_state_map(filtered_df)
        render_cost_analysis(filtered_df)
        render_model_performance(metrics, feature_importance)

    elif page == "🎯 Risk Scorer":
        render_risk_scorer()
    elif page == "🤖 AI Assistant":
        render_chat()

def render_chat():
    st.title("🤖 ReadmitIQ Assistant")
    st.caption("Ask questions about the readmission data in natural language")

    # initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "collection" not in st.session_state:
        st.session_state.collection = get_collection()

    # display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # chat input
    if prompt := st.chat_input("Ask about readmission rates, costs, demographics..."):
        # show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # get answer
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer, st.session_state.history, sources = ask(
                    prompt,
                    st.session_state.history,
                    st.session_state.collection
                )
            st.markdown(answer)
            st.caption(f"Sources: {', '.join(sources)}")

        st.session_state.messages.append({"role": "assistant", "content": answer})


@st.cache_resource
def load_model():
    import xgboost as xgb
    import pickle
    import json

    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgboost_readmission.json"))

    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    with open(MODELS_DIR / "label_maps.json") as f:
        label_maps = json.load(f)

    return model, encoders, feature_cols, label_maps


def render_risk_scorer():
    st.title("🎯 Patient Readmission Risk Scorer")
    st.caption("Enter patient characteristics to predict 30-day readmission risk")

    model, encoders, feature_cols, label_maps = load_model()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Patient Demographics")
        age_group = st.selectbox(
            "Age Group",
            options=label_maps.get("AGE_GROUP", ["Under 65", "65-74", "75-84", "85+"])
        )
        sex = st.selectbox(
            "Sex",
            options=label_maps.get("SEX", ["Male", "Female"])
        )
        race = st.selectbox(
            "Race",
            options=label_maps.get("RACE", ["White", "Black", "Hispanic", "Asian", "Other"])
        )
        state = st.selectbox(
            "State",
            options=sorted(label_maps.get("PRVDR_STATE_CD", ["NY", "CA", "TX"]))
        )

    with col2:
        st.markdown("#### Admission Details")
        admission_type = st.selectbox(
            "Admission Type",
            options=label_maps.get("ADMISSION_TYPE", ["Emergency", "Elective", "Urgent"])
        )
        dgns_category = st.selectbox(
            "Diagnosis Category",
            options=sorted(label_maps.get("DGNS_CATEGORY", ["Circulatory", "Respiratory"]))
        )
        discharge_status = st.selectbox(
            "Discharge Status",
            options=label_maps.get("DISCHARGE_STATUS", ["Home", "SNF", "Other"])
        )
        los_group = st.selectbox(
            "Length of Stay",
            options=label_maps.get("LOS_GROUP", ["1 day", "2-3 days", "4-7 days", "8-14 days", "15+ days"])
        )

    st.markdown("#### Financial Details")
    st.caption("💡 Tip: Average charges are ~$8,400 for non-readmitted and ~$2,700 for readmitted patients. Lower charges often indicate shorter, more acute stays.")
    col3, col4 = st.columns(2)
    col3, col4 = st.columns(2)
    with col3:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0, max_value=500000,
            value=6000, step=500
        )

        medicare_payment = st.number_input(
            "Medicare Payment ($)",
            min_value=0, max_value=200000,
            value=3000, step=500
        )
        los_days = st.number_input(
            "Length of Stay (days)",
            min_value=0, max_value=100,
            value=3, step=1
        )
    with col4:
        medicare_payment = st.number_input(
            "Medicare Payment ($)",
            min_value=0, max_value=500000,
            value=4000, step=500
        )

    cost_coverage = round(medicare_payment / total_charges, 4) if total_charges > 0 else 0

    st.divider()

    if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
        # build input dict
        raw_input = {
            "DGNS_CATEGORY": dgns_category,
            "ADMISSION_TYPE": admission_type,
            "DISCHARGE_STATUS": discharge_status,
            "AGE_GROUP": age_group,
            "SEX": sex,
            "RACE": race,
            "LOS_GROUP": los_group,
            "PRVDR_STATE_CD": state,
            "CLM_TOT_CHRG_AMT": total_charges,
            "CLM_PMT_AMT": medicare_payment,
            "CLM_UTLZTN_DAY_CNT": los_days,
            "COST_COVERAGE_RATIO": cost_coverage,
        }

        # encode categoricals
        encoded = {}
        for col in feature_cols:
            if col in encoders:
                val = raw_input[col]
                classes = list(encoders[col].classes_)
                if val in classes:
                    encoded[col] = encoders[col].transform([val])[0]
                else:
                    encoded[col] = 0
            else:
                encoded[col] = raw_input[col]

        # build dataframe in correct feature order
        import pandas as pd
        input_df = pd.DataFrame([encoded])[feature_cols]

        # predict
        risk_prob = model.predict_proba(input_df)[0][1]
        risk_pct = round(risk_prob * 100, 1)

        # display result
        st.divider()
        st.markdown("### Prediction Result")

        if risk_pct < 20:
            risk_level = "🟢 Low Risk"
            color = "#00aa44"
            recommendation = "Standard discharge planning. Schedule routine follow-up within 14 days."
        elif risk_pct < 40:
            risk_level = "🟡 Medium Risk"
            color = "#ffaa00"
            recommendation = "Enhanced discharge planning recommended. Follow-up call within 72 hours. Consider transitional care program."
        elif risk_pct < 60:
            risk_level = "🟠 High Risk"
            color = "#ff6600"
            recommendation = "High-intensity discharge planning required. Home health referral recommended. Follow-up within 48 hours."
        else:
            risk_level = "🔴 Very High Risk"
            color = "#cc0000"
            recommendation = "Consider extended stay or SNF placement. Immediate care coordinator assignment. Daily follow-up calls post-discharge."

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style="background:#1a1a1a; border-left: 6px solid {color};
                        padding: 2rem; border-radius: 8px; text-align:center;">
                <div style="font-size:3rem; font-weight:bold; color:{color}">
                    {risk_pct}%
                </div>
                <div style="font-size:1.2rem; color:{color}; margin-top:0.5rem">
                    {risk_level}
                </div>
                <div style="color:#aaa; margin-top:0.5rem; font-size:0.85rem">
                    30-Day Readmission Probability
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("#### Clinical Recommendation")
            st.info(recommendation)
            st.markdown("#### Key Risk Factors for This Patient")
            # show top factors based on feature importance
            fi = pd.read_csv(MODELS_DIR / "feature_importance.csv")
            top_features = fi.head(5)["feature"].tolist()
            for feat in top_features:
                display_val = raw_input.get(feat, "N/A")
                st.markdown(f"- **{feat}:** {display_val}")

        st.caption("⚠️ This prediction is based on synthetic Medicare data and is for demonstration purposes only. Not for clinical use.")

if __name__ == "__main__":
    main()