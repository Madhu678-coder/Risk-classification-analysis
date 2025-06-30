import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
 
# ---------------------
# 🧠 Preprocessing Function
# ---------------------
def preprocess(df):
    # Drop ID if present
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
 
    categorical_cols = ['Education', 'Securities Account', 'CD Account', 'Online', 'CreditCard', 'ZIP Code']
    numeric_cols = [col for col in df.columns if col not in categorical_cols + ['Personal Loan']]
 
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
 
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
 
    return df
 
 
# ---------------------
# 🎯 Streamlit App UI
# ---------------------
st.set_page_config(page_title="Loan Predictor", layout="centered")
st.title(" Bank Personal Loan Predictor")
 
st.markdown("Upload a CSV file with customer data to predict if a personal loan will be approved.")
 
# ---------------------
# 📤 Upload CSV
# ---------------------
uploaded_file = st.file_uploader(" Upload CSV File", type=["csv"])
 
# ---------------------
# 🤖 Load Trained Model
# ---------------------
model = joblib.load('best_final_model.pkl')
 
# ---------------------
# 🔄 Handle File Upload & Prediction
# ---------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
 
        # Backup original for display
        original_df = df.copy()
 
        # Apply preprocessing
        df_transformed = preprocess(df)
 
        # Predict
        preds = model.predict(df_transformed)
        original_df['Loan Prediction'] = ['Approved ✅' if p == 1 else 'Rejected ❌' for p in preds]
 
        # Display
        st.subheader(" Prediction Results")
        st.dataframe(original_df)
 
        # Download button
        csv = original_df.to_csv(index=False).encode('utf-8')
        st.download_button(" Download Results", csv, file_name="loan_predictions.csv", mime="text/csv")
 
    except Exception as e:
        st.error(f" Error: {e}")
else:
    st.info(" Please upload a CSV file.")
 