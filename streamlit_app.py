import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Workforce Intelligence Platform — Arav Mendiratta",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL STYLING
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* ---------- navigation radio buttons ---------- */
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.14);
        border-color: #7c3aed;
    }

    /* ---------- hero card ---------- */
    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-card h1 { font-size: 2.4rem; margin-bottom: 0.3rem; color: white; }
    .hero-card p  { font-size: 1.05rem; opacity: 0.92; }

    /* ---------- tech pill badges ---------- */
    .tech-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .tech-pill {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.82rem;
        font-weight: 500;
        color: white;
    }

    /* ---------- section explanation box ---------- */
    .section-explain {
        background: rgba(124, 58, 237, 0.08);
        border-left: 4px solid #7c3aed;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.4rem;
        line-height: 1.65;
    }

    /* ---------- architecture step cards ---------- */
    .arch-step {
        background: #f8f7ff;
        border: 1px solid #e8e4f9;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        height: 100%;
    }
    .arch-step .step-icon { font-size: 2rem; margin-bottom: 0.4rem; }
    .arch-step h4 { margin: 0.3rem 0; color: #4c1d95; }
    .arch-step p  { font-size: 0.88rem; color: #555; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA & MODEL (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/final_ml_dataset.csv')

@st.cache_resource
def load_model(_df):
    features = ['Age', 'Tenure', 'StressLevel', 'SelfRatedHealth', 'Sentiment_Score']
    X = _df[features].fillna(_df[features].median())
    y = _df['At_Risk']
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model

df = load_data()
model = load_model(df)

# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigate")
    page = st.radio(
        label="Choose a section:",
        options=[
            "🏠  Project Overview",
            "📊  Executive Dashboard",
            "🧠  NLP Sentiment Analysis",
            "🤖  Predictive ML Sandbox",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.78rem;opacity:0.55;text-align:center;'>"
        "Built by Arav Mendiratta<br>"
        "Python · SQL · ML · NLP · FastAPI · Docker"
        "</p>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
# PAGE: PROJECT OVERVIEW (Landing)
# ══════════════════════════════════════════════
if page.startswith("🏠"):
    # Hero banner
    st.markdown(
        """
        <div class="hero-card">
            <h1>🚀 Advanced Workforce Intelligence Platform</h1>
            <p>
                An end-to-end <strong>Data Science & Data Engineering</strong> portfolio project
                demonstrating production-grade skills across the full analytics lifecycle —
                from raw data generation and ETL, through Machine Learning and NLP,
                to interactive visualisation and containerised deployment.
            </p>
            <div class="tech-pills">
                <span class="tech-pill">Python</span>
                <span class="tech-pill">Pandas</span>
                <span class="tech-pill">SQL / SQLite</span>
                <span class="tech-pill">Scikit-Learn</span>
                <span class="tech-pill">TextBlob NLP</span>
                <span class="tech-pill">Streamlit</span>
                <span class="tech-pill">Plotly</span>
                <span class="tech-pill">FastAPI</span>
                <span class="tech-pill">Docker</span>
                <span class="tech-pill">GitHub Actions CI/CD</span>
                <span class="tech-pill">Pytest</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Motivation & Aim ──
    st.markdown("## 🎯 Motivation & Aim")
    st.markdown(
        """
        In large organisations, **employee wellbeing data is scattered** across HR systems,
        health surveys, and unstructured feedback forms — with no single source of truth.

        I built this project to solve that problem end-to-end:

        1. **Simulate a realistic HR data environment** — with messy, multi-source datasets
           containing duplicates, missing values, and inconsistent formatting.
        2. **Engineer a robust ETL pipeline** — using Python and SQL to clean, merge, and
           warehouse the data into a single analytical dataset.
        3. **Apply Machine Learning** — training a Random Forest classifier to **predict
           which employees are at risk** of severe work-ability decline, using demographic,
           health, and NLP-derived features.
        4. **Extract insight from unstructured text** — running NLP sentiment analysis on
           free-text employee survey feedback to quantify workplace morale.
        5. **Deliver results through two independent interfaces** — this interactive
           Streamlit dashboard *and* a standalone FastAPI REST microservice.
        6. **Follow production best practices** — Docker containerisation, automated Pytest
           testing, and a GitHub Actions CI/CD pipeline.
        """
    )

    # ── Architecture ──
    st.markdown("## 🏗️ Architecture & Pipeline")
    st.markdown("The platform follows a layered architecture. Each stage is a separate, runnable script:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="arch-step">'
            '<div class="step-icon">📂</div>'
            '<h4>1 · Data Generation</h4>'
            '<p>5,000 synthetic employees across 3 raw CSV datasets with intentional data-quality issues.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="arch-step">'
            '<div class="step-icon">🔄</div>'
            '<h4>2 · ETL & SQL</h4>'
            '<p>SQLite database pipeline with SQL JOINs, deduplication, imputation, and text standardisation.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="arch-step">'
            '<div class="step-icon">🧠</div>'
            '<h4>3 · NLP & ML</h4>'
            '<p>TextBlob sentiment scoring + Random Forest classifier trained with train/test evaluation.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="arch-step">'
            '<div class="step-icon">🚀</div>'
            '<h4>4 · Deployment</h4>'
            '<p>Streamlit dashboard, FastAPI microservice, Docker, Pytest, and GitHub Actions CI/CD.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── What you can explore ──
    st.markdown("---")
    st.markdown("## 👈 What You Can Explore")
    st.markdown(
        """
        Use the **sidebar** on the left to navigate between the three interactive modules:

        | Section | What It Shows | Skills Demonstrated |
        |---|---|---|
        | **📊 Executive Dashboard** | KPI metrics and interactive scatter plots of ABI scores by age, department, and demographics. | Pandas, Plotly, Streamlit, data storytelling |
        | **🧠 NLP Sentiment Analysis** | Distribution of NLP sentiment scores extracted from employee survey text, cross-referenced against stress levels. | TextBlob, NLP pipelines, box plots, correlation analysis |
        | **🤖 Predictive ML Sandbox** | A live inference form — adjust employee parameters and the Random Forest model predicts risk in real time. | Scikit-Learn, model serving, feature engineering, interactive ML |
        """
    )

    # ── Quick stats ──
    st.markdown("---")
    st.markdown("## 📈 Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{len(df):,}")
    c2.metric("Departments", df['Department'].nunique())
    c3.metric("Avg ABI Score", round(df['Calculated_ABI_Score'].mean(), 1))
    c4.metric("Employees At Risk", f"{int(df['At_Risk'].sum()):,}")


# ══════════════════════════════════════════════
# PAGE: EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════
elif page.startswith("📊"):
    st.markdown("# 📊 Executive Dashboard")

    st.markdown(
        '<div class="section-explain">'
        '<strong>What is this?</strong> &nbsp;A high-level KPI overview of the workforce, '
        'built to demonstrate <strong>data visualisation</strong> and <strong>analytical storytelling</strong> skills. '
        'The Work Ability Index (ABI) is a validated occupational-health metric that scores an employee\'s '
        'capacity to continue working. Scores range from 7 (poor) to 49 (excellent).<br><br>'
        '<strong>Skills demonstrated:</strong> Pandas aggregation, Plotly interactive charts, '
        'Streamlit layout & metrics, and data-driven narrative design.'
        '</div>',
        unsafe_allow_html=True,
    )

    # KPIs
    st.subheader("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", f"{len(df):,}")
    col2.metric("Average ABI Score", round(df['Calculated_ABI_Score'].mean(), 1))
    col3.metric("Employees At Risk", f"{int(df['At_Risk'].sum()):,}")

    # Scatter
    st.subheader("ABI Score by Age & Department")
    st.caption("Each dot is one employee. Colour encodes department; hover for details.")
    fig = px.scatter(
        df, x="Age", y="Calculated_ABI_Score", color="Department",
        opacity=0.6, hover_data=["Gender", "Location"],
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Age",
        yaxis_title="ABI Score",
        legend_title="Department",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: NLP SENTIMENT ANALYSIS
# ══════════════════════════════════════════════
elif page.startswith("🧠"):
    st.markdown("# 🧠 NLP Sentiment Analysis")

    st.markdown(
        '<div class="section-explain">'
        '<strong>What is this?</strong> &nbsp;Employee survey responses were processed through an '
        '<strong>NLP (Natural Language Processing)</strong> pipeline using the TextBlob library. '
        'Each piece of free-text feedback was scored on a polarity scale from <strong>−1.0</strong> (very negative) '
        'to <strong>+1.0</strong> (very positive). This converts unstructured qualitative data into a '
        'quantitative feature that the ML model can consume.<br><br>'
        '<strong>Why it matters:</strong> Sentiment is a leading indicator of burnout and disengagement — '
        'it often declines before hard metrics like sick days spike. By surfacing it early, HR teams can '
        'intervene proactively.<br><br>'
        '<strong>Skills demonstrated:</strong> TextBlob NLP, unstructured-to-structured feature engineering, '
        'Plotly histograms & box plots, cross-variable correlation analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Distribution of Sentiment Scores")
    st.caption("Histogram of polarity scores across all employees, coloured by department.")
    fig2 = px.histogram(
        df, x="Sentiment_Score", color="Department", nbins=50,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig2.update_layout(template="plotly_white", xaxis_title="Sentiment Polarity", yaxis_title="Count")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sentiment vs. Stress Level")
    st.caption("Box plots reveal how self-reported stress correlates with NLP-derived sentiment.")
    fig3 = px.box(
        df, x="StressLevel", y="Sentiment_Score", color="Department",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig3.update_layout(template="plotly_white", xaxis_title="Stress Level (1–10)", yaxis_title="Sentiment Score")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: PREDICTIVE ML SANDBOX
# ══════════════════════════════════════════════
elif page.startswith("🤖"):
    st.markdown("# 🤖 Predictive ML Sandbox")

    st.markdown(
        '<div class="section-explain">'
        '<strong>What is this?</strong> &nbsp;A live inference interface powered by a '
        '<strong>Scikit-Learn Random Forest Classifier</strong> trained on the full dataset. '
        'Adjust the sliders below to describe a hypothetical employee, then click '
        '<em>Predict Risk</em> to see the model\'s real-time prediction.<br><br>'
        '<strong>How the model works:</strong> The classifier was trained on five features — '
        'Age, Tenure, Stress Level, Self-Rated Health, and the NLP Sentiment Score — to predict '
        'whether an employee is <strong>at high risk</strong> of severe ABI decline or elevated '
        'absenteeism. The model was evaluated using an 80/20 train-test split with a full '
        'classification report (precision, recall, F1).<br><br>'
        '<strong>Skills demonstrated:</strong> Scikit-Learn model training & serialisation, '
        'feature engineering, real-time model serving, interactive ML prediction UI.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Employee Parameters")
    col1, col2 = st.columns(2)
    with col1:
        age_input = st.slider("Age", 20, 70, 35)
        tenure_input = st.slider("Tenure (Years)", 0, 30, 5)
        stress_input = st.slider("Stress Level (1–10)", 1, 10, 5)
    with col2:
        health_input = st.slider("Self-Rated Health (1–5)", 1, 5, 3)
        sentiment_input = st.slider("NLP Sentiment Score (−1.0 to 1.0)", -1.0, 1.0, 0.0)

    if st.button("🔮 Predict Risk", type="primary"):
        features = [[age_input, tenure_input, stress_input, health_input, sentiment_input]]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(
                f"⚠️ **HIGH RISK** — The model predicts this employee profile is at risk of "
                f"severe work-ability decline or high absenteeism.  \n"
                f"**Risk probability:** {probability:.1%}"
            )
        else:
            st.success(
                f"✅ **LOW RISK** — The model predicts this employee profile's work ability is stable.  \n"
                f"**Risk probability:** {probability:.1%}"
            )
