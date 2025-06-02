from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Expected feature columns based on the notebook
EXPECTED_COLUMNS = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Dependents_0', 'Dependents_1', 'Dependents_2',
    'Dependents_3+', 'Property_Area_Rural', 'Property_Area_Semiurban',
    'Property_Area_Urban', 'Gender', 'Married', 'Education', 'Self_Employed'
]

# prompt: Re-define các bước tiền xử lý dữ liệu đã thực hiện vào trong một method duy nhất, bao gồm cả việc load weights của Standard Scaler. Để tôi import vào một file code khác as a back-end.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class LoanPredictorPreprocessor:
    def __init__(self, scaler_path='standard_scaler.pkl'):
        """
        Initializes the preprocessor with the path to the saved StandardScaler.

        Args:
            scaler_path (str): Path to the saved Standard Scaler file.
        """
        self.scaler_path = scaler_path
        self.scaler = None

    def load_scaler(self):
        """Loads the pre-trained StandardScaler from the specified path."""
        try:
            self.scaler = joblib.load(self.scaler_path)
            print("StandardScaler loaded successfully.")
        except FileNotFoundError:
            print(f"Error: StandardScaler file not found at {self.scaler_path}")
            self.scaler = None # Ensure scaler is None if file not found
        except Exception as e:
            print(f"Error loading StandardScaler: {e}")
            self.scaler = None # Ensure scaler is None if loading fails

    def typecast(self, loan_data):
        """
        Typecasts the input DataFrame to ensure correct data types for each column.

        Args:
            loan_data (pd.DataFrame): The raw loan data as a pandas DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with typecasted columns.
            
        Desired type for each column:
        
        Data columns (total 13 columns):
        #   Column             Non-Null Count  Dtype  
        ---  ------             --------------  -----  
        0   Loan_ID            614 non-null    object 
        1   Gender             601 non-null    object 
        2   Married            611 non-null    object 
        3   Dependents         599 non-null    object 
        4   Education          614 non-null    object 
        5   Self_Employed      582 non-null    object 
        6   ApplicantIncome    614 non-null    int64  
        7   CoapplicantIncome  614 non-null    float64
        8   LoanAmount         592 non-null    float64
        9   Loan_Amount_Term   600 non-null    float64
        10  Credit_History     564 non-null    float64
        11  Property_Area      614 non-null    object 
        12  Loan_Status        614 non-null    object 

        """
        # Ensure the input is a DataFrame
        if not isinstance(loan_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        COLUMN_TYPES = {
            'Loan_ID': 'object',
            'Gender': 'object',
            'Married': 'object',
            'Dependents': 'object',
            'Education': 'object',
            'Self_Employed': 'object',
            'ApplicantIncome': 'int64',
            'CoapplicantIncome': 'float64',
            'LoanAmount': 'float64',
            'Loan_Amount_Term': 'float64',
            'Credit_History': 'float64',
            'Property_Area': 'object',
            # 'Loan_Status': 'object'
        }

        for (col, dtype) in COLUMN_TYPES.items():
            if dtype.startswith('float'):
                # Convert to float, handling NaNs
                loan_data[col] = pd.to_numeric(loan_data[col], errors='coerce')
            elif dtype.startswith('int'):
                # Convert to int, handling NaNs
                loan_data[col] = pd.to_numeric(loan_data[col], errors='coerce').fillna(-1000).astype(int)
        loan_data = loan_data.astype(COLUMN_TYPES, errors='ignore')

        if 'Loan_Status' in loan_data.columns:
            loan_data.drop(columns=['Loan_Status'], inplace=True)

        return loan_data
        

    def preprocess(self, loan_data):
        """
        Applies the preprocessing steps to the input DataFrame.

        Args:
            loan_data (pd.DataFrame): The raw loan data as a pandas DataFrame.

        Returns:
            np.ndarray: The preprocessed numerical features as a NumPy array.
                        Returns None if preprocessing fails (e.g., scaler not loaded).
        """
        if self.scaler is None:
            print("Scaler not loaded. Cannot preprocess data.")
            return None
        # Convert the list of dictionaries to a DataFrame
        if isinstance(loan_data, list):
            # If loan_data is a list of dictionaries, convert it to a DataFrame
            loan_data = pd.DataFrame(loan_data)
        elif isinstance(loan_data, dict):
            # If loan_data is a single dictionary, convert it to a DataFrame with one row
            loan_data = pd.DataFrame([loan_data])
        # Fix error: Error in prediction: Must pass 2-d input. shape=(1, 1, 12)
 
        loan_data = self.typecast(loan_data)
        print(loan_data.info())
 
        # --- Data Cleaning ---
        # Drop unneeded features
        if 'Loan_ID' in loan_data.columns:
            loan_data = loan_data.drop(['Loan_ID'], axis=1)

        # Handling Missing Values (using mode for categorical, mean for numerical)
        # Need to handle potential missing values for new data
        for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
            if col in loan_data.columns and loan_data[col].isnull().any():
                mode_val = loan_data[col].mode()
                if not mode_val.empty:
                  loan_data[col] = loan_data[col].fillna(mode_val[0])
                else:
                    # Handle case where mode is empty (e.g., all NaN)
                    print(f"Warning: Cannot fill missing values for {col} as mode is empty.")


        for col in ['LoanAmount', 'Loan_Amount_Term']:
             if col in loan_data.columns and loan_data[col].isnull().any():
                mean_val = loan_data[col].mean()
                if not pd.isna(mean_val):
                    loan_data[col] = loan_data[col].fillna(mean_val)
                else:
                     # Handle case where mean is NaN (e.g., all NaN)
                    print(f"Warning: Cannot fill missing values for {col} as mean is NaN.")


        # Encoding Categorical Variables
        loan_data = pd.get_dummies(loan_data)

        # Drop unneeded dummy features.
        # Ensure these columns exist before dropping
        cols_to_drop_dummies = []
        if 'Gender_Female' in loan_data.columns: 
            cols_to_drop_dummies.extend(['Gender_Female', 'Gender_'])
        if 'Married_No' in loan_data.columns: 
            cols_to_drop_dummies.extend(['Married_No', 'Married_'])
        if 'Education_Not Graduate' in loan_data.columns: cols_to_drop_dummies.append('Education_Not Graduate')
        if 'Self_Employed_No' in loan_data.columns: 
            cols_to_drop_dummies.extend(['Self_Employed_No', 'Self_Employed_'])
        if 'Dependents_' in loan_data.columns:
            cols_to_drop_dummies.append('Dependents_')
        # We don't drop Loan_Status_N here if we are preprocessing for prediction
        # If loan_data contains the target variable, drop it later before scaling.

        loan_data = loan_data.drop(columns=cols_to_drop_dummies, errors='ignore') # Use errors='ignore'

        # Rename existing columns
        newColunmsNames = {'Gender_Male': 'Gender',
                           'Married_Yes': 'Married',
                           'Education_Graduate': 'Education',
                           'Self_Employed_Yes': 'Self_Employed'} # Loan_Status_Y is the target, keep it if present

        loan_data.rename(columns=newColunmsNames, inplace=True)

        # --- Handling Outliers ---
        # This part from the original code seems to modify the dataframe in-place
        # by filtering rows based on IQR. This might not be suitable for a
        # general preprocessing function for new data where outliers should
        # likely be handled differently (e.g., capping or transformation).
        # For now, we will skip the outlier removal by filtering as it
        # changes the number of samples, which is usually done on training data.
        # If you need outlier handling, consider adding capping or transformation here.

        # --- Data Preprocessing ---
        # Separate features (X) if Loan_Status is present (for training/evaluation data)
        # If Loan_Status is not present (for new data prediction), just use the whole dataframe
        if 'Loan_Status_Y' in loan_data.columns:
             # This is likely the training/evaluation case where you have the target
             # We will return only the scaled features, assuming the target is handled separately
             X = loan_data.drop(columns=['Loan_Status_Y'])
        else:
            # This is likely the prediction case for new data
            X = loan_data.copy()
        print(X.columns)

        # Ensure consistent column order with the data the scaler was trained on
        # This requires knowing the column order from the training data.
        # A robust way is to save the list of columns after one-hot encoding
        # and before scaling during the training phase and use that list here.
        # For now, we'll assume the column order is consistent based on the get_dummies
        # but this is a potential point of failure if the columns in new data differ.
        # A better approach would be to save the columns list used for training X.

        # Data Re-scaling and Normalizing the features
        # Apply the loaded scaler
        print(X)
        # return X
        try:
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=X.columns)
        except Exception as e:
            print(f"Error during scaling: {e}")
            return None

# Example usage:
# Assuming you have a new pandas DataFrame named 'new_loan_data'

# Create an instance of the preprocessor, specifying the path to the saved scaler
loan_preprocessor = LoanPredictorPreprocessor(scaler_path='standard_scaler.pkl') # Replace with your actual scaler path
loan_preprocessor.load_scaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:   
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Handle both single row and multiple rows
        if isinstance(data, list):
            print("RECEIVED MULTIPLE ROWS")
            predictions = []
            processed_data = loan_preprocessor.preprocess(data)
            # print("PROCESSED DATA:", processed_data.head())
            preds = model.predict(processed_data)
            probs = model.predict_proba(processed_data) if hasattr(model, 'predict_proba') else None
            # print(f"FINISHED INFERENCE: {preds[:5]}, {probs[:5]}")
            for i, row in enumerate(data):
                if processed_data is None:
                    predictions.append({'error': 'Error processing row'})
                    continue
                
                prediction = preds[i]
                prob = probs[i].max() if probs is not None else None
                
                result = {
                    'prediction': 'Approved' if prediction == 0 else 'Rejected',
                    'prediction_value': int(prediction),
                    'confidence': float(prob) if prob is not None else None
                }
                predictions.append(result)
            
            '''
            for row in data:
                processed_row = loan_preprocessor.preprocess(row)
                if processed_row is None:
                    predictions.append({'error': 'Error processing row'})
                    continue
                      # Convert to DataFrame and make prediction
                df = pd.DataFrame([processed_row])
                
                # Note: The model was trained on scaled data, but we can't recreate the exact scaler
                # For now, predict without scaling to see if it works better
                prediction = model.predict(df)[0]
                probability = model.predict_proba(df)[0].max() if hasattr(model, 'predict_proba') else None
                
                result = {
                    'prediction': 'Approved' if prediction == 1 else 'Rejected',
                    'prediction_value': int(prediction),
                    'confidence': float(probability) if probability else None
                }
                predictions.append(result)
            '''
            print(f"Predictions: {predictions}")
            return jsonify({'predictions': predictions})
        else:
            processed_row = LoanPredictorPreprocessor.preprocess(data)
            if processed_row is None:
                return jsonify({'error': 'Error processing data'}), 400
                
            df = pd.DataFrame([processed_row])
            
            # Apply MinMax scaling
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df)
            
            prediction = model.predict(df_scaled)[0]
            probability = model.predict_proba(df_scaled)[0].max() if hasattr(model, 'predict_proba') else None
            
            print( f"Prediction: {prediction}, Probability: {probability}")

            result = {
                'prediction': 'Approved' if prediction == 1 else 'Rejected',
                'prediction_value': int(prediction),
                'confidence': float(probability) if probability else None
            }
            
            return jsonify(result)
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
