# CreditWise — AI-Powered Loan Intelligence System

CreditWise is a full-stack Django web application that integrates Machine Learning to automate loan approval decisions. The system analyzes applicant financial details and predicts whether a loan application should be Approved or Rejected, along with the probability of approval.

The prediction engine uses a **Decision Tree classifier (max depth 15)** trained on 975,000+ realistic Indian financial records. It was selected after benchmarking multiple models (Logistic Regression, KNN, Naive Bayes, Decision Tree) and tuning tree depth on a held-out test set.

## Features

- **Loan Approval Predictor** — Enter your financial profile and get real-time approval probability with animated visual feedback
- **Key Factor Analysis** — See how your credit score, DTI ratio, collateral, and existing loans affect your chances
- **Improvement Tips** — Personalized suggestions to improve approval odds
- **EMI Calculator** — Calculate monthly EMI with three interest methods: Reducing Balance, Flat Rate, and Compound Interest
- **Amortization Schedule** — Year-by-year repayment breakdown with donut chart visualization
- **DTI Auto-Calculator** — Automatically calculates your Debt-to-Income ratio from monthly income and debt inputs

---

## Tech Stack

| Layer             | Technology                    |
| ----------------- | ----------------------------- |
| Backend           | Python, Django                |
| ML Model          | Scikit-learn (Decision Tree)  |
| Data Processing   | Pandas, NumPy                 |
| Frontend          | HTML, CSS, Vanilla JavaScript |
| Styling           | Custom dark luxury theme      |
| Model Persistence | Joblib                        |
| Deployment        | Render, Gunicorn, WhiteNoise  |

---

## Machine Learning Model

- **Final Algorithm** — Decision Tree Classifier (`max_depth=15`, `min_samples_leaf=50`)
- **Dataset Size** — 975,800 processed records (80% train / 20% test split, `random_state=42`)
- **Test Accuracy** — 87.1%
- **Test F1 Score** — 77.4%
- **Train Accuracy** — 88.8% (small train-test gap, so the tree is not heavily overfitting)

### Model Selection

Rather than committing to one algorithm up front, I benchmarked several models on the same train/test split and compared them on accuracy and F1 score.

| Model                        | Accuracy  | F1 Score  |
| ---------------------------- | --------- | --------- |
| Gaussian Naive Bayes         | 71.5%     | 63.9%     |
| K-Nearest Neighbors (k=7)    | 78.8%     | 60.1%     |
| Logistic Regression          | 83.1%     | 69.3%     |
| **Decision Tree (depth 15)** | **87.1%** | **77.4%** |

Logistic Regression, KNN and Naive Bayes were trained on standardized features. The Decision Tree was trained on raw features, since trees do not need scaling.

### Decision Tree Depth Tuning

I then tuned the tree's `max_depth` to find the point where test performance stops improving:

| Max Depth | Train Acc | Test Acc  | F1 Score  |
| --------- | --------- | --------- | --------- |
| 3         | 79.2%     | 79.1%     | 59.6%     |
| 5         | 82.0%     | 82.0%     | 68.4%     |
| 7         | 84.3%     | 84.0%     | 71.9%     |
| 10        | 87.0%     | 86.3%     | 75.5%     |
| **15**    | **88.8%** | **87.1%** | **77.4%** |
| 20        | 88.9%     | 87.0%     | 77.3%     |
| None      | 88.9%     | 87.0%     | 77.3%     |

Depth 15 was chosen because test accuracy and F1 stop improving beyond it, so a deeper tree adds complexity without any gain.

### Why Decision Tree?

- Highest accuracy and F1 score among all tested models
- Captures non-linear relationships and feature interactions (for example, a high loan amount combined with a low credit score) that Logistic Regression cannot
- No feature scaling required, which makes the prediction pipeline simpler
- Interpretable: feature importances show which factors drive each decision

### Features Used (27 total)

- Applicant Income, Co-applicant Income, Age, Dependents
- Credit Score (squared), DTI Ratio (squared), Collateral Ratio
- Savings, Loan Amount, Loan Term, Existing Loans
- Employment Status, Employer Category, Education Level
- Marital Status, Gender, Loan Purpose, Property Area

### Top Feature Importances (Decision Tree)

| Feature          | Importance |
| ---------------- | ---------- |
| Credit Score²    | 0.247      |
| Loan Amount      | 0.226      |
| DTI Ratio²       | 0.190      |
| Applicant Income | 0.119      |
| Existing Loans   | 0.083      |
| Collateral Ratio | 0.077      |
| Savings          | 0.032      |

### Key Correlations with Approval

| Feature               | Correlation |
| --------------------- | ----------- |
| Credit Score²         | +0.38       |
| Employment (Salaried) | +0.27       |
| Collateral Ratio      | +0.16       |
| DTI Ratio²            | -0.31       |
| Existing Loans        | -0.19       |
| Loan Amount           | -0.13       |

---

## Project Structure

```
CreditWiseLoanSystem/
├── ML/
│   ├── Data/
│   │   ├── data_Preprocessing.ipynb
│   │   ├── loan_approval_data.csv
│   │   └── Processed_loan_approval_data.csv
│   └── Train and Test Model/
│       ├── Train_test_Model.ipynb   # model comparison and depth tuning
│       └── final_model.py           # trains and saves the final model
├── model/
│   └── loan_model.pkl
├── web/
│   ├── manage.py
│   ├── settings.py
│   ├── urls.py
│   ├── views.py
│   ├── wsgi.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
├── test.py                          # quick sanity check of the saved model
├── Procfile
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Anaconda (recommended)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/mayankchouhan263/CreditWiseLoanSystem.git
cd CreditWiseLoanSystem
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the development server**

```bash
cd web
python manage.py runserver
```

**4. Open in browser**

```
http://127.0.0.1:8000
```

---

## Retraining the Model

If you want to retrain with new data:

```bash
# Step 1 — Run the preprocessing notebook
# Open ML/Data/data_Preprocessing.ipynb and run all cells
# This generates ML/Data/Processed_loan_approval_data.csv

# Step 2 — (Optional) Compare models
# Open "ML/Train and Test Model/Train_test_Model.ipynb"
# to benchmark Logistic Regression, KNN, Naive Bayes and Decision Tree

# Step 3 — Train and save the final model (run from the project root)
python "ML/Train and Test Model/final_model.py"

# Step 4 — Sanity check
python test.py

# Step 5 — Restart the server
cd web
python manage.py runserver
```

---

## Input Ranges

The model accepts real Indian rupee values:

| Field            | Range                   |
| ---------------- | ----------------------- |
| Monthly Income   | ₹10,000 – ₹1,00,00,000  |
| Loan Amount      | ₹10,000 – ₹50,00,00,000 |
| Savings          | ₹1,000 – ₹10,00,00,000  |
| Collateral Value | ₹0 – ₹1,00,00,00,000    |
| Credit Score     | 300 – 900               |
| DTI Ratio        | 0.05 – 0.90             |

---

## 📊 Dataset

The dataset used to train this model is publicly available on Kaggle:

🔗 **[CreditWise Loan Approval Dataset — Kaggle](https://www.kaggle.com/datasets/mayankchouhan263/loan-approval-datasetrealistic-indian-rupee-data)**

- 1,000,000 synthetic Indian loan applications
- Realistic Indian rupee ranges (₹10,000 – ₹1,00,00,000 income)
- 13 features including credit score, DTI ratio, collateral, employment status
- Target: `Loan_Approved` (1 = Approved, 0 = Rejected)

---

## Deployment

This project is configured for deployment on **Render**.

### Environment Variables

| Key             | Value                        |
| --------------- | ---------------------------- |
| `SECRET_KEY`    | Your secret key              |
| `DEBUG`         | `False`                      |
| `ALLOWED_HOSTS` | `your-app-name.onrender.com` |

### Build Command

```bash
pip install -r requirements.txt && cd web && python manage.py collectstatic --noinput
```

### Start Command

```bash
cd web && gunicorn wsgi:application --bind 0.0.0.0:$PORT --timeout 120 --workers 1
```

---

## Author

**Mayank** — Built as a full-stack ML project combining data science, Django backend, and frontend UI design.

---

## License

This project is for educational purposes.
