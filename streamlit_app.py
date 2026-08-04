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
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# SESSION STATE (navigation)
# ──────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page_name):
    st.session_state.page = page_name

# ──────────────────────────────────────────────
# GLOBAL STYLING
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ---------- hide default sidebar ---------- */
    [data-testid="stSidebar"] { display: none; }

    /* ---------- hero card ---------- */
    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2.2rem 2rem 1.8rem 2rem;
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero-card h1 { font-size: 2.2rem; margin-bottom: 0.3rem; color: white; }
    .hero-card p  { font-size: 1rem; opacity: 0.92; margin-bottom: 0.2rem; }

    /* ---------- tech pill badges ---------- */
    .tech-pills { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
    .tech-pill {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 500;
        color: white;
    }

    /* ---------- navigation cards ---------- */
    .nav-card-dashboard {
        background: linear-gradient(145deg, #0f172a 0%, #1e3a5f 100%);
        border: 1px solid rgba(59,130,246,0.3);
    }
    .nav-card-dashboard:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 30px rgba(59,130,246,0.3);
    }
    .nav-card-dashboard h3 { color: #93c5fd !important; }
    .nav-card-dashboard .card-skills { color: #60a5fa !important; }

    .nav-card-nlp {
        background: linear-gradient(145deg, #1a0a2e 0%, #2d1654 100%);
        border: 1px solid rgba(168,85,247,0.3);
    }
    .nav-card-nlp:hover {
        border-color: #a855f7;
        box-shadow: 0 8px 30px rgba(168,85,247,0.3);
    }
    .nav-card-nlp h3 { color: #d8b4fe !important; }
    .nav-card-nlp .card-skills { color: #c084fc !important; }

    .nav-card-ml {
        background: linear-gradient(145deg, #0a2818 0%, #134e2a 100%);
        border: 1px solid rgba(34,197,94,0.3);
    }
    .nav-card-ml:hover {
        border-color: #22c55e;
        box-shadow: 0 8px 30px rgba(34,197,94,0.3);
    }
    .nav-card-ml h3 { color: #86efac !important; }
    .nav-card-ml .card-skills { color: #4ade80 !important; }

    .nav-card-dashboard, .nav-card-nlp, .nav-card-ml {
        border-radius: 16px;
        padding: 1.5rem 1.3rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .nav-card-dashboard:hover, .nav-card-nlp:hover, .nav-card-ml:hover {
        transform: translateY(-5px);
    }
    .nav-card-dashboard .card-icon,
    .nav-card-nlp .card-icon,
    .nav-card-ml .card-icon { font-size: 2.2rem; margin-bottom: 0.4rem; }

    .nav-card-dashboard h3,
    .nav-card-nlp h3,
    .nav-card-ml h3 { margin: 0.3rem 0; font-size: 1.1rem; }

    .nav-card-dashboard p,
    .nav-card-nlp p,
    .nav-card-ml p { font-size: 0.85rem; color: rgba(255,255,255,0.7); line-height: 1.5; margin: 0.4rem 0 0 0; }

    .nav-card-dashboard .card-skills,
    .nav-card-nlp .card-skills,
    .nav-card-ml .card-skills {
        margin-top: 0.6rem;
        font-size: 0.73rem;
        font-weight: 600;
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

    /* ---------- CTA header ---------- */
    .cta-header {
        text-align: center;
        margin: 0.8rem 0 0.3rem 0;
    }
    .cta-header h2 { color: #4c1d95; margin-bottom: 0.1rem; font-size: 1.5rem; }
    .cta-header p  { color: #666; font-size: 0.95rem; }
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


# ══════════════════════════════════════════════
# PAGE: HOME (Landing)
# ══════════════════════════════════════════════
if st.session_state.page == "home":
    # Hero banner
    st.markdown(
        """
        <div class="hero-card">
            <h1>🚀 Advanced Workforce Intelligence Platform</h1>
            <p>
                An end-to-end <strong>Data Science & Data Engineering</strong> portfolio project —
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

    # ── NAVIGATION CARDS — right after hero, no scrolling ──
    st.markdown(
        '<div class="cta-header">'
        '<h2>Explore the Interactive Modules</h2>'
        '<p>Click any card below to dive into the live analysis.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        st.markdown(
            '<div class="nav-card-dashboard">'
            '<div class="card-icon">📊</div>'
            '<h3>Executive Dashboard</h3>'
            '<p>KPI metrics and interactive scatter plots of ABI scores across age, department, and location.</p>'
            '<div class="card-skills">Pandas · Plotly · Streamlit · Data Storytelling</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Open Executive Dashboard →", key="btn_dashboard", use_container_width=True):
            go_to("dashboard")
            st.rerun()

    with nav2:
        st.markdown(
            '<div class="nav-card-nlp">'
            '<div class="card-icon">🧠</div>'
            '<h3>NLP Sentiment Analysis</h3>'
            '<p>Sentiment scores from employee survey text, cross-referenced with stress levels.</p>'
            '<div class="card-skills">TextBlob · NLP Pipelines · Correlation Analysis</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Open NLP Analysis →", key="btn_nlp", use_container_width=True):
            go_to("nlp")
            st.rerun()

    with nav3:
        st.markdown(
            '<div class="nav-card-ml">'
            '<div class="card-icon">🤖</div>'
            '<h3>Predictive ML Sandbox</h3>'
            '<p>Adjust employee parameters and the Random Forest model predicts risk in real time.</p>'
            '<div class="card-skills">Scikit-Learn · Feature Engineering · Model Serving</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Open ML Sandbox →", key="btn_ml", use_container_width=True):
            go_to("ml")
            st.rerun()

    # ── Motivation & Aim (below the cards) ──
    st.markdown("---")
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

    # ── Dataset at a Glance ──
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
elif st.session_state.page == "dashboard":
    if st.button("← Back to Project Overview"):
        go_to("home")
        st.rerun()

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
elif st.session_state.page == "nlp":
    if st.button("← Back to Project Overview"):
        go_to("home")
        st.rerun()

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
elif st.session_state.page == "ml":
    if st.button("← Back to Project Overview"):
        go_to("home")
        st.rerun()

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
