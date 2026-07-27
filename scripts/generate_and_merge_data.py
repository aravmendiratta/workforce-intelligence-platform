import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

NUM_EMPLOYEES = 5000

print(f"Generating synthetic data for {NUM_EMPLOYEES} employees...")

# --- Dataset 1: Employee Demographics ---
employee_ids = [f"EMP{str(i).zfill(5)}" for i in range(1, NUM_EMPLOYEES + 1)]
ages = np.random.randint(22, 65, size=NUM_EMPLOYEES)
# Intentional messiness in Gender for cleansing step
genders_raw = np.random.choice(['Male', 'Female', 'M', 'F', 'male', 'female', 'Other'], size=NUM_EMPLOYEES, p=[0.4, 0.4, 0.05, 0.05, 0.04, 0.04, 0.02])
departments = np.random.choice(['Sales', 'IT', 'HR', 'Operations', 'Finance', 'Marketing'], size=NUM_EMPLOYEES)
tenures = np.random.randint(1, 25, size=NUM_EMPLOYEES)
locations = np.random.choice(['New York', 'London', 'Berlin', 'Tokyo', 'Remote'], size=NUM_EMPLOYEES)

demographics_df = pd.DataFrame({
    'EmployeeID': employee_ids,
    'Age': ages,
    'Gender': genders_raw,
    'Department': departments,
    'Tenure': tenures,
    'Location': locations
})

demographics_df.to_csv('data/raw/demographics.csv', index=False)
print("Saved data/raw/demographics.csv")

# --- Dataset 2: Health Survey Results ---
# Introduce some missing values intentionally
self_rated_health = np.random.choice([1, 2, 3, 4, 5], size=NUM_EMPLOYEES, p=[0.05, 0.15, 0.4, 0.3, 0.1])
chronic_conditions = np.random.choice(['Yes', 'No'], size=NUM_EMPLOYEES, p=[0.25, 0.75])
sick_days = np.random.poisson(lam=4, size=NUM_EMPLOYEES)
# Add correlation: higher stress in IT, and missing values
stress_levels = []
for dept in departments:
    if dept == 'IT':
        stress = np.random.randint(6, 11)  # 6-10
    else:
        stress = np.random.randint(1, 10)
    
    # 5% chance of missing value
    if random.random() < 0.05:
        stress_levels.append(np.nan)
    else:
        stress_levels.append(stress)

health_survey_df = pd.DataFrame({
    'EmployeeID': employee_ids,
    'SelfRatedHealth': self_rated_health,
    'ChronicConditions': chronic_conditions,
    'SickDaysPastYear': sick_days,
    'StressLevel': stress_levels
})

# Add some duplicate rows for cleansing
health_survey_df = pd.concat([health_survey_df, health_survey_df.sample(n=50)])
health_survey_df = health_survey_df.sample(frac=1).reset_index(drop=True)

health_survey_df.to_csv('data/raw/health_survey.csv', index=False)
print("Saved data/raw/health_survey.csv")

# --- Dataset 3: Work Ability Index Questionnaire ---
# Correlation: IT has lower Work Ability due to stress
current_work_ability = []
work_ability_physical = np.random.choice([1, 2, 3, 4, 5], size=NUM_EMPLOYEES)
work_ability_mental = []

for dept in departments:
    if dept == 'IT':
        current_work_ability.append(np.random.randint(2, 8))
        work_ability_mental.append(np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))
    else:
        current_work_ability.append(np.random.randint(4, 11))
        work_ability_mental.append(np.random.choice([2, 3, 4, 5]))

future_work_ability = np.random.choice(['Yes', 'No'], size=NUM_EMPLOYEES, p=[0.9, 0.1])

abi_responses_df = pd.DataFrame({
    'EmployeeID': employee_ids,
    'CurrentWorkAbility': current_work_ability,
    'WorkAbilityPhysical': work_ability_physical,
    'WorkAbilityMental': work_ability_mental,
    'FutureWorkAbilityEstimate': future_work_ability
})

abi_responses_df.to_csv('data/raw/abi_responses.csv', index=False)
print("Saved data/raw/abi_responses.csv")

# ==========================================
# STEP 2: DATA CLEANSING & MERGING
# ==========================================
print("\n--- Starting Data Cleansing & Merging ---")

# 1. Load and Merge
print("Merging datasets on EmployeeID...")
merged_df = pd.merge(demographics_df, health_survey_df, on='EmployeeID', how='inner')
merged_df = pd.merge(merged_df, abi_responses_df, on='EmployeeID', how='inner')

# 2. Cleanse
# Remove duplicates
initial_len = len(merged_df)
merged_df = merged_df.drop_duplicates(subset='EmployeeID')
print(f"Removed {initial_len - len(merged_df)} duplicate rows.")

# Handle missing values
# Impute missing StressLevel with department median
print("Imputing missing StressLevel with department median...")
merged_df['StressLevel'] = merged_df.groupby('Department')['StressLevel'].transform(lambda x: x.fillna(x.median()))

# Standardize text for Gender
print("Standardizing Gender text...")
gender_map = {
    'Male': 'M', 'male': 'M', 'M': 'M',
    'Female': 'F', 'female': 'F', 'F': 'F',
    'Other': 'Other'
}
merged_df['Gender'] = merged_df['Gender'].map(gender_map)

# 3. Export
output_file = 'data/processed/master_health_data.xlsx'
print(f"Exporting cleansed data to {output_file}...")
try:
    merged_df.to_excel(output_file, index=False)
    print("Export successful!")
except ModuleNotFoundError:
    print("Warning: openpyxl module not found. Please install it with 'pip install openpyxl' to save as Excel.")
    print("Saving as data/processed/master_health_data.csv instead.")
    merged_df.to_csv('data/processed/master_health_data.csv', index=False)

print("Data generation and merging complete!")
