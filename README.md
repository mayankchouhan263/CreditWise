# CreditWise — AI-Powered Loan Intelligence System

CreditWise is a full-stack Django web application that integrates Machine Learning to automate loan approval decisions. The system analyzes applicant financial details and predicts whether a loan application should be Approved or Rejected, along with the probability of approval.

The prediction engine uses a Logistic Regression model trained on 975,000+ realistic Indian financial records, enabling data-driven decision making similar to real banking risk assessment systems.

## Features

- **Loan Approval Predictor** — Enter your financial profile and get real-time approval probability with animated visual feedback
- **Key Factor Analysis** — See how your credit score, DTI ratio, collateral, and existing loans affect your chances
- **Improvement Tips** — Personalized suggestions to improve approval odds
- **EMI Calculator** — Calculate monthly EMI with three interest methods: Reducing Balance, Flat Rate, and Compound Interest
- **Amortization Schedule** — Year-by-year repayment breakdown with donut chart visualization
- **DTI Auto-Calculator** — Automatically calculates your Debt-to-Income ratio from monthly income and debt inputs

---

## Tech Stack

| Layer             | Technology                         |
| ----------------- | ---------------------------------- |
| Backend           | Python, Django                     |
| ML Model          | Scikit-learn (Logistic Regression) |
| Data Processing   | Pandas, NumPy                      |
| Frontend          | HTML, CSS, Vanilla JavaScript      |
| Styling           | Custom dark luxury theme           |
| Model Persistence | Joblib                             |

---

## Machine Learning Model

- **Algorithm** — Logistic Regression
- **Training Data** — 975,800 rows of synthetic Indian loan data
- **Accuracy** — 83.1%
- **Precision** — 74.8%
- **Recall** — 64.6%
- **F1 Score** — 69.3%

### Features Used (27 total)

- Applicant Income, Co-applicant Income, Age, Dependents
- Credit Score (squared), DTI Ratio (squared), Collateral Ratio
- Savings, Loan Amount, Loan Term, Existing Loans
- Employment Status, Employer Category, Education Level
- Marital Status, Gender, Loan Purpose, Property Area

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
│   │   ├── loan_approval_data.csv
│   │   └── Processed_loan_approval_data.csv
│   └── Train and Test Model/
│       └── final_model.py
├── model/
│   ├── loan_model.pkl
│   └── scaler.pkl
└── web/
    ├── manage.py
    ├── settings.py
    ├── urls.py
    ├── views.py
    ├── wsgi.py
    ├── templates/
    │   └── index.html
    └── static/
        ├── style.css
        └── script.js
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Anaconda (recommended)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/CreditWiseLoanSystem.git
cd CreditWiseLoanSystem
```

**2. Install dependencies**

```bash
pip install django scikit-learn pandas numpy joblib whitenoise gunicorn
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
# Open ML/data_Preprocessing.ipynb and run all cells
# This generates ML/Data/Processed_loan_approval_data.csv

# Step 2 — Train the model
python "ML/Train and Test Model/final_model.py"

# Step 3 — Restart the server
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
cd web && gunicorn wsgi:application --bind 0.0.0.0:$PORT
```

---

## Author

**Mayank** — Built as a full-stack ML project combining data science, Django backend, and frontend UI design.

---

## License

This project is for educational purposes.
