import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Advanced HR Intelligence", layout="wide")

# Load Data and Model
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

st.title("🚀 Advanced Workforce Intelligence Platform")
st.markdown("An end-to-end Machine Learning, NLP, and Data Analytics Dashboard built with Python, Pandas, SQL, and Streamlit.")

tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🧠 NLP Sentiment Analysis", "🤖 Predictive ML Sandbox"])

# --- TAB 1: Dashboard ---
with tab1:
    st.header("Work Ability Index (ABI) Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df))
    col2.metric("Average ABI Score", round(df['Calculated_ABI_Score'].mean(), 1))
    col3.metric("Employees At Risk", int(df['At_Risk'].sum()))
    
    st.subheader("ABI Score by Age & Department")
    fig = px.scatter(df, x="Age", y="Calculated_ABI_Score", color="Department", opacity=0.6,
                     hover_data=["Gender", "Location"])
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: NLP Analysis ---
with tab2:
    st.header("Employee Feedback NLP Sentiment")
    st.markdown("We extracted unstructured text from employee surveys and ran **TextBlob** sentiment analysis to quantify emotion.")
    
    fig2 = px.histogram(df, x="Sentiment_Score", color="Department", nbins=50, title="Distribution of Sentiment Scores")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Sentiment vs. Stress Level")
    fig3 = px.box(df, x="StressLevel", y="Sentiment_Score", color="Department")
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 3: Predictive ML ---
with tab3:
    st.header("Random Forest Predictive Sandbox")
    st.markdown("Input employee parameters below to let the **Scikit-Learn Random Forest** model predict if they are at high risk of severe Work Ability decline.")
    
    col1, col2 = st.columns(2)
    with col1:
        age_input = st.slider("Age", 20, 70, 35)
        tenure_input = st.slider("Tenure (Years)", 0, 30, 5)
        stress_input = st.slider("Stress Level (1-10)", 1, 10, 5)
    with col2:
        health_input = st.slider("Self-Rated Health (1-5)", 1, 5, 3)
        sentiment_input = st.slider("NLP Sentiment Score (-1.0 to 1.0)", -1.0, 1.0, 0.0)
    
    if st.button("Predict Risk"):
        features = [[age_input, tenure_input, stress_input, health_input, sentiment_input]]
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            st.error("⚠️ HIGH RISK: The model predicts this employee is at risk of severe work ability decline or high absence.")
        else:
            st.success("✅ LOW RISK: The model predicts this employee's work ability is stable.")
