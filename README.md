# Customer_Churn_Prediction
A comprehensive analysis and model development pipeline for predicting customer churn using machine learning techniques. The project includes data preprocessing, SMOTE for balancing, feature engineering, and model evaluation with Random Forest and XGBoost algorithms."

# Customer Churn Prediction

This project focuses on predicting customer churn using machine learning techniques. By using a dataset from Kaggle, we implement various data preprocessing steps, visualizations, and model-building methods to identify key factors contributing to customer churn.

## Project Overview
- **Source**: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Techniques Used**: SMOTE for handling class imbalance, feature importance analysis, and model evaluation metrics.
- **Models Used**: Random Forest and XGBoost

## Key Steps
1. **Data Preprocessing**: Convert data types, handle missing values, and engineer new features.
2. **Exploratory Data Analysis**: Visualize churn distribution, tenure breakdown, and average charges.
3. **Class Imbalance Handling**: Use SMOTE to balance the dataset, ensuring fair training for both classes.
4. **Model Training**: Train and evaluate Random Forest and XGBoost models.
5. **Feature Importance Analysis**: Visualize top features that impact churn prediction.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)

### Setup
1. Clone the repository:
   https://github.com/dachib04/Customer_Churn_Prediction
2. Create a virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install dependencies:
   pip install -r requirements.txt


## Dependencies
- `imbalanced-learn`
- `shap`
- `seaborn`
- `matplotlib`
- `sklearn`

## Results
Model performance is measured in terms of Accuracy, Precision, Recall, F1 Score, and AUC. The top 10 features influencing customer churn are visualized.

## Usage
Clone the repository and run `churn_prediction.ipynb` to reproduce results. Ensure you download the dataset and adjust the path accordingly.

