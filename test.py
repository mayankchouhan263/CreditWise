import joblib, numpy as np

model = joblib.load("model/loan_model.pkl")

# Good profile — high income, good credit, low DTI
good = [[100000, 0, 35, 1, 0, 300000, 2000000, 60, 1,
         1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
         2.5, 0.0625, 490000]]

# Bad profile — low income, bad credit, high DTI
bad  = [[15000, 0, 45, 3, 5, 10000, 5000000, 360, 0,
         0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
         0.002, 0.3136, 90000]]

for label, row in [("Good", good), ("Bad", bad)]:
    prob = model.predict_proba(np.array(row))[0][1]
    print(f"{label}: {round(prob * 100)}%")