# Customer_Churn_Prediction
A comprehensive analysis and model development pipeline for predicting customer churn using machine learning techniques. The project includes data preprocessing, SMOTE for balancing, feature engineering, and model evaluation with Random Forest and XGBoost algorithms."

# Customer Churn Prediction

This project focuses on predicting customer churn using machine learning techniques. By using a dataset from Kaggle, we implement various data preprocessing steps, visualizations, and model-building methods to identify key factors contributing to customer churn.

## Project Overview
- **Data**: Customer churn data from Kaggle (not included due to size, but can be downloaded https://www.kaggle.com/datasets/blastchar/telco-customer-churn.
- **Models Used**: Random Forest and XGBoost
- **Techniques**: SMOTE for handling class imbalance, feature importance analysis, and model evaluation metrics.

## Key Steps
1. **Data Preprocessing**: Convert data types, handle missing values, and engineer new features.
2. **Exploratory Data Analysis**: Visualize churn distribution, tenure breakdown, and average charges.
3. **Class Imbalance Handling**: Use SMOTE to balance the dataset, ensuring fair training for both classes.
4. **Model Training**: Train and evaluate Random Forest and XGBoost models.
5. **Feature Importance Analysis**: Visualize top features that impact churn prediction.

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

