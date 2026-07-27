import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("--- Starting ML & NLP Training Pipeline ---")

# 1. Load Data
df = pd.read_csv('data/processed/advanced_master_data.csv')

# 2. NLP Sentiment Analysis
print("Running NLP Sentiment Analysis on Employee Feedback...")
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['Sentiment_Score'] = df['Employee_Feedback'].apply(get_sentiment)

# 3. Define Target Variable for ML (At-Risk Employee)
# At risk if ABI is < 37 (Poor/Moderate) OR Sick Days > 8
print("Defining Target Variable (At_Risk)...")
df['At_Risk'] = np.where((df['Calculated_ABI_Score'] < 37) | (df['SickDaysPastYear'] > 8), 1, 0)
print(f"Total At-Risk Employees: {df['At_Risk'].sum()} out of {len(df)}")

# 4. Feature Engineering for ML
features = ['Age', 'Tenure', 'StressLevel', 'SelfRatedHealth', 'Sentiment_Score']
X = df[features]
y = df['At_Risk']

# Impute any remaining NaNs just in case
X = X.fillna(X.median())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest Model
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
preds = rf_model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc:.2f}")
print("Classification Report:")
print(classification_report(y_test, preds))

# 6. Save the model and final dataset
joblib.dump(rf_model, 'models/rf_model.pkl')
df.to_csv('data/processed/final_ml_dataset.csv', index=False)
print("Model saved to models/rf_model.pkl and data saved to data/processed/final_ml_dataset.csv!")
