from flask import Flask, render_template, request
import numpy as np
import joblib

# Load trained model
model = joblib.load('best_final_model.pkl')

# Define model accuracy (update this with your actual value from training)
model_accuracy = 0.88  # 88% accuracy

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from form
        input_values = [float(x) for x in request.form.values()]

        # Validate input feature count
        if len(input_values) != model.n_features_in_:
            return render_template(
                'index1.html',
                prediction=f"❌ Expected {model.n_features_in_} features but got {len(input_values)}."
            )

        final_input = [input_values]

        # Predict
        prediction = model.predict(final_input)[0]
        result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'

        return render_template(
            'index1.html',
            prediction=result,
            accuracy=f"{model_accuracy * 100:.2f}%"
        )

    except Exception as e:
        return render_template('index1.html', prediction=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
