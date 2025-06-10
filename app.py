from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'nsl_kdd_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Create the Flask app
app = Flask(__name__)

# Load the feature template used during training (column names, dummies)
# You should save this during model training. For now, use a placeholder:
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))  # <-- Save and load this in your train script

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Example: input values from form
        input_dict = request.form.to_dict()
        
        # Convert input to a DataFrame
        input_df = pd.DataFrame([input_dict])

        # Convert numeric strings to actual numbers
        for col in input_df.columns:
            try:
                input_df[col] = input_df[col].astype(float)
            except:
                pass  # If it fails, it's likely categorical

        # One-hot encode categorical features to match training
        input_df = pd.get_dummies(input_df)

        # Align with training columns (fill missing with 0, reorder)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f"Predicted Threat Type: {prediction}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

#if __name__ == "__main__":
 #   app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)
