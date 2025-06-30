from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
 
# Load model (could be pipeline with scaler inside)
model = joblib.load('best_final_model.pkl')
 
app = Flask(__name__)
UPLOAD_FOLDER='uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
   
    if not file:
        return "No file uploaded."
 
    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
 
    # Read data
    df = pd.read_csv(filepath)
    df.drop(columns=["ID"],inplace=True)
 
    # Predict
    predictions = model.predict(df)
    df['Loan Prediction'] = ['Approved ✅' if p == 1 else 'Rejected ❌' for p in predictions]
 
    # Show predictions as HTML table
    return df.to_html(classes='table table-bordered')
 
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER,exist_ok=True)
    app.run(debug=True)