# Advanced Workforce Intelligence Platform 🚀

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-TextBlob-green.svg)](https://textblob.readthedocs.io/)

An end-to-end Data Science and Engineering pipeline designed to process, analyze, and predict employee health metrics and the Work Ability Index (ABI).

**Live Dashboard:** [Link to your Streamlit App Here]

## 📌 Project Overview
This project simulates a real-world HR analytics environment. It features:
1. **Automated ETL Pipeline:** A local SQLite database pipeline extracting and transforming unstructured and structured employee data.
2. **Machine Learning:** A Scikit-Learn Random Forest Classifier trained to predict "At-Risk" employees based on demographics, stress levels, and historical health data.
3. **Natural Language Processing (NLP):** Sentiment analysis on unstructured employee survey feedback to quantify workplace morale.
4. **Interactive Dashboard:** A custom Streamlit web application providing live ML predictions and Plotly visualizations.

## 🏗️ Architecture & Files
- `database_etl.py`: Generates the SQLite database and executes the SQL JOIN pipeline.
- `train_ml_model.py`: Runs NLP sentiment extraction and trains the Random Forest classifier, exporting `rf_model.pkl`.
- `streamlit_app.py`: The frontend interactive dashboard.
- `automate_excel.py` & `automate_presentation.py`: Python automation scripts used to programmatically generate Excel pivot tables and PowerPoint slide decks.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/workforce-intelligence-platform.git
   cd workforce-intelligence-platform
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Re-run the Pipeline:**
   ```bash
   python database_etl.py
   python train_ml_model.py
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🧠 Machine Learning Model
The Random Forest model predicts if an employee is highly likely to experience a severe decline in their Work Ability Index. It relies on features such as Age, Tenure, Stress Level, Self-Rated Health, and an NLP-derived Sentiment Score.
