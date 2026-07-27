import pandas as pd
import sqlite3
import random
import os

print("--- Starting Advanced ETL Pipeline ---")

# 1. Load existing data
print("Loading raw CSVs...")
demographics = pd.read_csv('data/raw/demographics.csv')
health = pd.read_csv('data/raw/health_survey.csv')
abi = pd.read_csv('data/raw/abi_responses.csv')

# 2. Generate Synthetic NLP Text Data (Employee Feedback)
print("Generating synthetic unstructured employee feedback...")
positive_phrases = ["Love the culture here.", "Great work-life balance.", "Management is very supportive.", "I feel energized at work.", "Excellent benefits."]
neutral_phrases = ["Work is okay.", "Standard corporate environment.", "Nothing special to report.", "It pays the bills.", "Average experience."]
negative_phrases = ["Extremely stressed and burnt out.", "Management expects too much.", "I am exhausted every day.", "Terrible work-life balance.", "I need a long break."]

feedback_list = []
for index, row in health.iterrows():
    stress = row['StressLevel']
    if pd.isna(stress):
        feedback_list.append(random.choice(neutral_phrases))
    elif stress >= 8:
        feedback_list.append(random.choice(negative_phrases))
    elif stress <= 4:
        feedback_list.append(random.choice(positive_phrases))
    else:
        feedback_list.append(random.choice(neutral_phrases))

health['Employee_Feedback'] = feedback_list

# 3. Create SQLite Database and load tables
db_path = 'data/database/hr_database.db'
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
print(f"Connected to local SQLite database: {db_path}")

demographics.to_sql('demographics', conn, index=False)
health.to_sql('health_survey', conn, index=False)
abi.to_sql('abi_responses', conn, index=False)

# 4. Perform SQL ETL Transformation
print("Executing SQL JOIN and transformations...")
sql_query = """
    SELECT 
        d.EmployeeID, d.Age, d.Gender, d.Department, d.Tenure, d.Location,
        h.SelfRatedHealth, h.ChronicConditions, h.SickDaysPastYear, h.StressLevel, h.Employee_Feedback,
        a.CurrentWorkAbility, a.WorkAbilityPhysical, a.WorkAbilityMental,
        -- Calculate ABI Score in SQL
        (a.CurrentWorkAbility * 2) + (a.WorkAbilityPhysical * 3) + (a.WorkAbilityMental * 2) + 4 AS Calculated_ABI_Score
    FROM demographics d
    INNER JOIN health_survey h ON d.EmployeeID = h.EmployeeID
    INNER JOIN abi_responses a ON d.EmployeeID = a.EmployeeID
"""

etl_df = pd.read_sql_query(sql_query, conn)

# Basic Cleansing in Pandas
etl_df = etl_df.drop_duplicates(subset='EmployeeID')
etl_df['StressLevel'] = etl_df.groupby('Department')['StressLevel'].transform(lambda x: x.fillna(x.median()))

# 5. Export to CSV for the ML layer
etl_df.to_csv('data/processed/advanced_master_data.csv', index=False)
print("ETL Pipeline complete! Saved data/processed/advanced_master_data.csv")
conn.close()
