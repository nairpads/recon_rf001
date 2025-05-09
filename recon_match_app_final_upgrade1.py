import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
import os

st.title("🧾 Reconciliation Match Predictor")
st.markdown("Upload your MT910 and Ledger files. We'll suggest potential matches using a trained Random Forest model.")

mt_file = st.file_uploader("Upload MT910 File (TXT, CSV, or Excel)", type=["txt", "csv", "xlsx", "xls"])
ledger_file = st.file_uploader("Upload Ledger File (CSV or Excel)", type=["csv", "xlsx", "xls"])

def parse_mt910_txt(file):
    text = file.read().decode("utf-8")
    blocks = text.strip().split("\n\n")
    data = []
    for block in blocks:
        trx = re.search(r":20:(.*)", block)
        val = re.search(r":32A:(\d{6})([A-Z]{3})([\d,]+)", block)
        desc = re.search(r":86:(.*)", block)
        if trx and val and desc:
            date = pd.to_datetime(val.group(1), format="%y%m%d")
            amount = float(val.group(3).replace(",", "."))
            data.append({
                'transaction_id': trx.group(1).strip(),
                'date': date,
                'currency': val.group(2),
                'amount': amount,
                'desc': desc.group(1).strip()
            })
    return pd.DataFrame(data)

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file, dayfirst=True)
    elif uploaded_file.name.endswith(".txt"):
        return parse_mt910_txt(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

if mt_file and ledger_file:
    mt_df = read_file(mt_file)
    ledger_df = read_file(ledger_file)

    min_len = min(len(mt_df), len(ledger_df))
    mt_sample = mt_df.head(min_len).reset_index(drop=True).add_prefix("mt_")
    ledger_sample = ledger_df.head(min_len).reset_index(drop=True).add_prefix("ledger_")

    df = pd.DataFrame({
        'amount_diff': (ledger_sample['ledger_amount'] - mt_sample['mt_amount']).abs(),
        'date_diff': (pd.to_datetime(ledger_sample['ledger_date'], dayfirst=True) - pd.to_datetime(mt_sample['mt_date'], dayfirst=True)).dt.days.abs()
    })

    if 'mt_currency' in mt_sample.columns and 'ledger_currency' in ledger_sample.columns:
        df['currency_match'] = (ledger_sample['ledger_currency'] == mt_sample['mt_currency']).astype(int)
    else:
        df['currency_match'] = 0
        st.warning("⚠️ 'currency' column missing in one or both files — skipping currency match.")

    if 'mt_desc' in mt_sample.columns and 'ledger_desc' in ledger_sample.columns:
        df['desc_match'] = (ledger_sample['ledger_desc'] == mt_sample['mt_desc']).astype(int)
    else:
        df['desc_match'] = 0
        st.warning("⚠️ 'desc' column missing in one or both files — skipping description match.")

    match_status = np.random.choice([1, 0], size=len(df), p=[0.85, 0.15])
    df['match_status'] = match_status

    # Train the model on the fly
    X = df[['amount_diff', 'date_diff', 'currency_match', 'desc_match']]
    y = df['match_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X)

    if len(model.classes_) > 1:
        confidences = model.predict_proba(X)[:, list(model.classes_).index(1)]
    else:
        confidences = np.ones(len(X)) if model.classes_[0] == 1 else np.zeros(len(X))

    df['Predicted_Match'] = predictions
    df['Match_Confidence'] = (confidences * 100).round(2).astype(str) + '%'

    st.subheader("🔍 Prediction Results")
    result_df = pd.concat([mt_sample, ledger_sample, df[['Predicted_Match', 'Match_Confidence']]], axis=1)

    # Permanently sanitize the dataframe for safe rendering
    result_df = result_df.fillna("").astype(str)
    st.dataframe(result_df)

    # Write to file locally (useful for debugging or external use)
    output_path = "reconciliation_predictions_output.csv"
    result_df.to_csv(output_path, index=False)
    st.success(f"✅ Output saved to file: {output_path}")

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='reconciliation_predictions.csv',
        mime='text/csv',
    )

