
## 1 — Adult / Census Income (UCI / Kaggle)
- Domain: Sociology, Public policy, Marketing  
- Task: Binary classification — predict if income > $50K/year  
- Instances: 48,842  
- Features: 14 mixed (age, workclass, education, occupation, race, sex, capital-gain, hours-per-week, etc.)  
- Description: Classic Census dataset for demographic-based income prediction. Good for testing preprocessing, categorical encoding, fairness analysis, and baseline classifiers.  
- Source: https://www.kaggle.com/datasets/uciml/adult-census-income

---

## 2 — Diabetes 130-US Hospitals for Years 1999–2008
- Domain: Health, Medicine  
- Task: Multi-class classification — predict readmission (<30 days, >30 days, no)  
- Instances: 101,766  
- Features: ~50 attributes (patient demographics, vitals, diagnoses, procedures, medications, hospital identifiers, admission details)  
- Description: Longitudinal, multi-hospital electronic health record data. Useful for temporal modeling, feature engineering for clinical events, and evaluating imbalanced/multi-class methods. Check de-identification and use policies before sharing.  
- Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

---

## 3 — Online Shoppers Purchasing Intention
- Domain: Marketing, E‑commerce  
- Task: Binary classification — predict whether a session ends in a purchase (revenue)  
- Instances: 12,330  
- Features: 10 numerical, 8 categorical, 5 date-related (page visit counts/durations, bounce/exit rates, product/traffic info)  
- Description: Session-level features from e-commerce site navigation. Great for session modeling, feature hashing, time-based features, and conversion-rate prediction experiments.  
- Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Notes/tips:
- Inspect class balance and missing values before modeling.  
- Consider standard preprocessing: one-hot or target encoding for categoricals, scaling for numeric features, and stratified splits for evaluation.  
- Cite original sources and check licensing if redistributing data.

