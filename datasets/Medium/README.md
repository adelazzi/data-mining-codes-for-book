1. Bank Marketing (Marketing)
- Domain: Marketing, Banking, Business Analytics
- Task: Binary classification — predict whether a client will subscribe to a term deposit (Yes/No)
- Description: Cleaned version of a direct-marketing dataset from a Portuguese bank. Records represent client contacts and campaign outcomes; typical preprocessing removes duplicates and noisy entries.
- Size (cleaned/sample): ~3,000 (original dataset is larger)
- Features: ~16 mixed (age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, etc.)
- CSV Link (Kaggle): https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

2. Default of Credit Card Clients (Finance)
- Domain: Finance, Banking, Risk Management
- Task: Binary classification — predict whether a client will default on next month’s payment
- Description: Well-known, structured dataset with demographic, credit, payment history, and billing information (Taiwan, 2005). Often downsampled/cleaned for experiments.
- Size (cleaned/sample): ~4,000 (original ~30,000)
- Features: 24 (demographic + credit & payment history)
    - Demographic: gender, education, marital status, age
    - Payment history / amounts: credit limit, Pay_0–Pay_6, Bill_Amt1–Bill_Amt6, Pay_Amt1–Pay_Amt6
- CSV Link (Kaggle): https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

3. Diabetes Health Indicators (Health)
- Domain: Public Health, Medicine
- Task: Binary classification — predict diabetes / high risk for diabetes
- Description: Cleaned sample from the CDC BRFSS survey (commonly used as the "diabetes_binary" benchmark). Contains demographic, behavioral, and health indicators.
- Size (cleaned/sample): ~3,000 (original 50k+)
- Features: ~21 (demographic, health metrics, and risk factors)
    - Examples: age, education, income, BMI, physical/mental health, physical activity, high BP, high cholesterol, smoking, stroke history, heart disease, heavy drinking, access to healthcare, difficulty walking, sex
- CSV Link (Kaggle): https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv
