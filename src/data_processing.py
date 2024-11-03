from src.imports import pd, zipfile, pd, np


def load_and_extract_data(file_path):
  # Extracts data from a zip file and loads the CSV file into a DataFrame
  with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall('telco_churn')
  data = pd.read_csv("telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

  # Display basic information about the data
  print(data.head())
  print(data.info())
  print(data.isnull().sum())
  return data

def process_data(data):
  # Convert TotalCharges to numeric and fill missing values
  data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
  # data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
  data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())


  # Avoid division by zero in AvgChargesPerService calculation
  data['tenure'] = data['tenure'].replace(0, np.nan)
  data['AvgChargesPerService'] = data['MonthlyCharges'] / data['tenure']

  # Bin tenure into brackets
  data['tenure_bracket'] = pd.cut(data['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=['0-12', '13-24', '25-36', '37-48', '49-60', '60+'])

  # Convert categorical variables to dummy variables
  data = pd.get_dummies(data, drop_first=True)

  # Separate numeric and categorical columns
  numeric_columns = data.select_dtypes(include=['float64', 'int']).columns
  categorical_columns = data.select_dtypes(include=['category', 'object']).columns

  # Fill NaN values in numeric columns with mean
  data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

  # Fill NaN values in categorical columns with the most frequent value
  data[categorical_columns] = data[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))

  # Replace infinite values in case there are any left
  data.replace([np.inf, -np.inf], np.nan, inplace=True)
  data.fillna(data.mean(), inplace=True)  # Final check to fill any remaining NaN

  return data

  
