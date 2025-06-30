from flask import Flask, render_template, request
import numpy as np
import joblib

# Load trained model (only)
model = joblib.load('best_final_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all input values from the form
        input_values = [float(x) for x in request.form.values()]

        # Check if input count matches model's expected input
        if len(input_values) != model.n_features_in_:
            return render_template(
                'index1.html',
                prediction=f"❌ Expected {model.n_features_in_} features but got {len(input_values)}."
            )

        # No scaler — use raw input
        final_input = [input_values]

        # Predict using model
        prediction = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][int(prediction)]

        result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'

        return render_template(
            'index1.html',
            prediction=result,
            probability=f"{prob * 100:.2f}%"
        )

    except Exception as e:
        return render_template('index1.html', prediction=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
