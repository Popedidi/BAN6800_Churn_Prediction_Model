# app.py
from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('Churn_Prediction_Model2.pkl')

# Automatically get expected features if the model supports it
if hasattr(model, 'feature_names_in_'):
    expected_features = model.feature_names_in_.tolist()
else:
    # Fallback to a predefined list if feature names are not available
    expected_features = [
        'Gender', 
        'Partner', 
        'Age', 
        'Tenure Months', 
        'Tech Support', 
        'Monthly Charge', 
        'Services', 
        'CLTV', 
        'City_Acton', 
        'City_Adelanto', 
        'City_Adin', 
        'City_Agoura Hills', 
        'City_Aguanga'
    ]

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict_individual', methods=['POST'])
def predict_individual():
    if request.method == 'POST':
        # Extract individual customer data from the form
        customer_data = {}
        
       # Populate customer_data with available features
        for feature in expected_features:
            if feature in request.form:
                customer_data[feature] = float(request.form[feature]) if feature != 'Services' else request.form.getlist('Services')
            else:
                # If the feature is not provided, set it to a default value (e.g., 0)
                customer_data[feature] = 0  # or any other default value as appropriate

        # Create DataFrame with the correct feature names
        customer_df = pd.DataFrame([customer_data])

        # Ensure the DataFrame has the same columns as the model expects
        customer_df = customer_df.reindex(columns=expected_features, fill_value=0)  # Fill missing columns with default values

         # Make prediction
        try:
            prediction = model.predict(customer_df)
            prediction_result = 'Churn' if prediction[0] == 1 else 'No Churn'
            return render_template('upload.html', prediction=prediction_result)
        except Exception as e:
            return render_template('upload.html', error=f"Error during prediction: {str(e)}")


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if request.method == 'POST':
        try:
            if request.files['file']:
                file = request.files['file']
                if not file or not file.filename.endswith('.csv'):
                    return render_template('upload.html', error="Please upload a valid CSV file."), 400

                data = pd.read_csv(file)

                # Create a DataFrame with default values for missing features
            for feature in expected_features:
                if feature not in data.columns:
                    data[feature] = 0  # Set default value for missing features
                    
                # Reindex the DataFrame to ensure it has the expected features
                data = data.reindex(columns=expected_features, fill_value=0)  # Fill missing columns with default values

                # Make predictions
                predictions = model.predict(data)
                data['Predictions'] = ['Churn' if pred == 1 else 'No Churn' for pred in predictions]

                # Convert predictions to an HTML table
                predictions_html = data.to_html(classes="table table-striped", index=False)
                return render_template('upload.html', predictions=predictions_html)

        except Exception as e:
            return render_template('upload.html', error=f"Error during batch prediction: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)