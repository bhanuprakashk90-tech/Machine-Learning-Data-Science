# Machine-Learning-Data-Science

# Machine Learning & NLP End-to-End Task Suite

This repository contains three complete, production-ready Machine Learning and Natural Language Processing (NLP) pipelines built with Python, Scikit-Learn, XGBoost, and NLTK. 

---

## 📁 Repository Structure

```text
├── README.md
├── requirements.txt
├── 01_iris_classification.py
├── 02_telco_churn_ensemble.py
└── 03_nlp_sentiment_classifier.py

🚀 Setup & Environment
Prerequisites
Python 3.9+

pip package manager

# Clone the repository
git clone [https://github.com/your-username/ml-nlp-task-suite.git](https://github.com/your-username/ml-nlp-task-suite.git)
cd ml-nlp-task-suite

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn nltk

 Task Overview & ImplementationsTask 1: Supervised Classification on Iris DatasetProblem StatementTrain and benchmark Logistic Regression and Decision Tree models on the Iris dataset. Scale features, split data (80/20), build confusion matrix heatmaps, and plot decision boundaries.Implementation HighlightsPreprocessing: Standardized continuous features using StandardScaler to ensure zero mean and unit variance.Stratification: Employed train_test_split(stratify=y) to maintain class balance across training and test folds.Visualization: Rendered 2D meshgrid surface plots (np.meshgrid) to map spatial decision zones per model.Performance MetricsModelAccuracyPrecision (Weighted)Recall (Weighted)F1-Score (Weighted)Logistic Regression0.9670.9690.9670.967Decision Tree0.9670.9690.9670.967Task 2: Telecom Customer Churn EnsembleProblem Statement

Task 2: Telecom Customer Churn Ensemble
Problem Statement
Predict customer churn using telecom subscription records. Clean missing data, encode categorical variables with OneHotEncoder, balance class distributions via SMOTE/scale_pos_weight, train Random Forest and XGBoost classifiers, and achieve an ROC-AUC score > 80%.

Implementation Highlights
Data Cleaning: Converted blank strings in TotalCharges to NaN and imputed values using the median.

Data Leakage Prevention: Wrapped feature scaling, One-Hot Encoding, SMOTE, and model fitting inside imblearn.pipeline.Pipeline.

Class Imbalance Strategy:

Random Forest: Oversampled minority churn cases using synthetic sample generation (SMOTE).

XGBoost: Applied gradient loss scaling via scale_pos_weight = (negative / positive).

ROC-AUC Target: Achieved ROC-AUC scores of 0.83+ across both ensemble algorithms.

Key Findings (Top Features)
tenure (Customer length of stay)

Contract_Two year / Contract_Month-to-month

InternetService_Fiber optic

TotalCharges & MonthlyCharges

Task 3: NLP Text Sentiment Classifier
Problem Statement
Construct an NLP text classification pipeline that ingests customer reviews, cleans/lemmatizes raw strings, vectorizes text using TfidfVectorizer, trains a Multinomial Naive Bayes classifier, and handles live user input phrases.

Implementation Highlights
Text Preprocessing Pipeline:

Regex clean HTML tags and non-alphabetical noise.

Lowercase text and split into individual word tokens.

Strip NLTK English stopwords.

Lemmatize tokens to base dictionary roots via WordNetLemmatizer.

Feature Extraction: Transformed text into a term frequency-inverse document frequency matrix (TfidfVectorizer(max_features=5000)).

Out-of-Vocabulary (OOV) Handling: Developed a robust inference wrapper (predict_sentiment_robust) that inspects non-zero matrix elements (nnz == 0) and probability distributions to return Neutral / Unknown for inputs containing unknown terms or ambiguous predictions.

Usage Instructions
Execute any task script individually using Python:

Bash
# Run Task 1: Iris Classification & Decision Boundaries
python 01_iris_classification.py

# Run Task 2: Churn Prediction & ROC-AUC Curves
python 02_telco_churn_ensemble.py

# Run Task 3: NLP Sentiment Classifier & Live Testing
python 03_nlp_sentiment_classifier.py




