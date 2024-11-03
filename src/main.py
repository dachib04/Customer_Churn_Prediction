from src.data_processing import load_and_extract_data, process_data
from src.model import split_and_balance_data, train_and_evaluate_model
from src.visualization import visulize_data, feature_importance_visualization
import os

# Define the path to your dataset
file_path = '/content/drive/My Drive/telco-customer-churn.zip'

def main():
  # Load and extract data from the zip file
  data = load_and_extract_data(file_path)
  print(data.head()) # Display first few rows of the data
  print(data.info()) # Display data types and non-null counts
  print(data.isnull().sum()) # Display count of missing values per column

  # Visualize initial data distribution
  visulize_data(data)
  # Preprocess data (handle missing values, encode categorical variables, etc.)
  data = process_data(data)
  # Split data into training and test sets and apply SMOTE to balance classes
  X_train, X_test, y_train, y_test = split_and_balance_data(data)
  # Train the model and evaluate its performance on the test set
  best_rf_model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
  # Visualize feature importance for the trained model
  feature_importance_visualization(best_rf_model, X_train)

if __name__ == "__main__":
    main()
