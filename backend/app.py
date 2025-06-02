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

def preprocess_data(data_row):
    """
    Preprocess a single row of data to match the model's expected format
    """
    # Create a dictionary with all expected columns initialized to 0
    processed_row = {col: 0 for col in EXPECTED_COLUMNS}
      # Map the input data to the expected format
    try:        # Helper function to safely convert to float
        def safe_float(value, default):
            if value is None or value == '' or value == 'nan' or str(value).lower() == 'nan':
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
          # Numerical features - handle missing values with defaults
        processed_row['ApplicantIncome'] = safe_float(data_row.get('ApplicantIncome'), 5000)
        processed_row['CoapplicantIncome'] = safe_float(data_row.get('CoapplicantIncome'), 2200)
        processed_row['LoanAmount'] = safe_float(data_row.get('LoanAmount'), 100)
        processed_row['Loan_Amount_Term'] = safe_float(data_row.get('Loan_Amount_Term'), 360)
        processed_row['Credit_History'] = safe_float(data_row.get('Credit_History'), 1.0)
        
        # Apply square root transformation (as done in the notebook)
        # Add small epsilon to avoid sqrt of 0
        processed_row['ApplicantIncome'] = np.sqrt(max(processed_row['ApplicantIncome'], 5000))
        processed_row['CoapplicantIncome'] = np.sqrt(max(processed_row['CoapplicantIncome'], 2200))
        processed_row['LoanAmount'] = np.sqrt(max(processed_row['LoanAmount'], 100))
          # Handle Dependents (one-hot encoding)
        dependents = str(data_row.get('Dependents', '0')).strip()
        if dependents in ['0', '1', '2', '3+']:
            processed_row[f'Dependents_{dependents}'] = 1
        else:
            processed_row['Dependents_0'] = 1 
            
        property_area = str(data_row.get('Property_Area', 'Semiurban')).strip()
        if property_area in ['Rural', 'Semiurban', 'Urban']:
            processed_row[f'Property_Area_{property_area}'] = 1
        else:
            processed_row['Property_Area_Semiurban'] = 1  

        gender = str(data_row.get('Gender', 'Male')).strip().lower()
        processed_row['Gender'] = 1 if gender == 'male' else 0
        
        married = str(data_row.get('Married', 'Yes')).strip().lower()
        processed_row['Married'] = 1 if married in ['yes', 'y'] else 0
        
        education = str(data_row.get('Education', 'Graduate')).strip().lower()
        processed_row['Education'] = 1 if education == 'graduate' else 0
        
        self_employed = str(data_row.get('Self_Employed', 'No')).strip().lower()
        processed_row['Self_Employed'] = 1 if self_employed in ['yes', 'y'] else 0
        
        return processed_row
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print(f"Data row: {data_row}")
        return None

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
            predictions = []
            for row in data:
                processed_row = preprocess_data(row)
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
            
            return jsonify({'predictions': predictions})
        else:
            processed_row = preprocess_data(data)
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
