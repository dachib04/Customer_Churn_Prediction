from src.imports import sns, plt



def visulize_data(data):
  """
    Visualizes the distribution of the 'Churn' variable and the distribution of 'tenure' by 'Churn' status.
    #so the parameter is - data (DataFrame): The dataset containing the 'Churn' and 'tenure' columns.
    """
  
   # Plot the distribution of the Churn variable
  sns.countplot(data['Churn'])
  plt.title('hurn Distribution')
  plt.show()

  # Plot the distribution of tenure by Churn status
  sns.histplot(data=data, x='tenure', hue='Churn', multiple='stack')
  plt.title('Tenure Distribution by Churn')
  plt.show()


def feature_importance_visualization(best_rf_model, X_train):

  """
    Visualizes the top 10 most important features as determined by the Random Forest model.

    Parameters are foollowing: 
    - best_rf_model (RandomForestClassifier): The trained Random Forest model.
    - X_train (DataFrame): The training dataset containing the feature columns.
    """

   # Create a DataFrame of feature importances
  rf_feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

  # Plot the top 10 important features for Random Forest
  plt.figure(figsize=(10, 6))
  sns.barplot(x='Importance', y='Feature', data=rf_feature_importances.head(10))
  plt.title('Top 10 Feature Importances - Random Forest')
  plt.show()
