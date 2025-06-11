from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and feature column structure
model = pickle.load(open("nsl_kdd_model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# In-memory prediction logs for dashboard
prediction_logs = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input as dictionary
        input_dict = request.form.to_dict()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Try casting numeric values
        for col in input_df.columns:
            try:
                input_df[col] = input_df[col].astype(float)
            except:
                pass  # Keep as string if categorical

        # One-hot encode categorical values
        input_df = pd.get_dummies(input_df)

        # Align with training features
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        # Log the prediction
        prediction_logs.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input': input_dict,
            'prediction': prediction
        })

        # Display result on form page
        return render_template('index.html', prediction_text=f"Predicted Threat Type: {prediction}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

@app.route('/dashboard')
def dashboard():
    counts = Counter([log['prediction'] for log in prediction_logs])
    return render_template('dashboard.html',
                           logs=prediction_logs[::-1],
                           counts=dict(counts),
                           total=len(prediction_logs))

if __name__ == "__main__":
    app.run(debug=True)
