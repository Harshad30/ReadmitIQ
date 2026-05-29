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
        # readmission rate by age group
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
        # readmission rate by diagnosis category
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
        # length of stay distribution
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
        # admission type breakdown
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
        ["📊 Dashboard", "🤖 AI Assistant"],
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

if __name__ == "__main__":
    main()