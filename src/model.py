from src.data_processing import load_and_extract_data, process_data
from src.imports import SMOTE, train_test_split, RandomForestClassifier, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def split_and_balance_data(data):
  
  # Separate the features and target variable from the data
  X = data.drop('Churn_Yes', axis=1)
  y = data['Churn_Yes']
  
  # Apply SMOTE to balance the dataset by oversampling the minority class
  smote = SMOTE(random_state=42)
  X_balanced, y_balanced = smote.fit_resample(X, y)
  
  # Split the balanced data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

  return X_train, X_test, y_train, y_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
  
  # Initialize and train a Random Forest model
  best_rf_model = RandomForestClassifier(random_state=42)
  best_rf_model.fit(X_train, y_train)
  
  # Make predictions on the test set
  y_pred_rf = best_rf_model.predict(X_test)
  y_proba_rf = best_rf_model.predict_proba(X_test)[:, 1]
  
  # Calculate performance metrics for the model
  accuracy_rf = accuracy_score(y_test, y_pred_rf)
  precision_rf = precision_score(y_test, y_pred_rf)
  recall_rf = recall_score(y_test, y_pred_rf)
  f1_rf = f1_score(y_test, y_pred_rf)
  auc_rf = roc_auc_score(y_test, y_proba_rf)
  
  # Print model performance metrics
  print("Random Forest Model Performance:")
  print("Accuracy:", accuracy_rf)
  print("Precision:", precision_rf)
  print("Recall:", recall_rf)
  print("F1 Score:", f1_rf)
  print("AUC:", auc_rf)

  return best_rf_model
