import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

# Patch custom functions if needed
def ordinal_encode_func(df): return df
sys.modules['__main__'].ordinal_encode_func = ordinal_encode_func

# Layout settings
st.set_page_config(page_title="📊 Telecom Churn App", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('Churn_data.csv')

# Load model, scaler, and model_columns from the pickle file
@st.cache_resource
def load_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        model, scaler, model_columns = pickle.load(f)
    return model, scaler, model_columns

# Load everything
data = load_data()
model, scaler, model_columns = load_model()

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Churn Prediction", "📈 Insights & Graphs", "📄 Raw Data"])

# ================== 🏠 MAIN PAGE: CHURN PREDICTION ==================
if page == "🏠 Churn Prediction":
    st.title("🔮 Telecom Churn Prediction")
    st.markdown("Enter important customer details to predict churn likelihood.")

    # Use only most relevant features
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider('Tenure (months)', 0, 100, 12)
        monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
        total = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)
    with col2:
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        payment = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

    # Build user input
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        f'Contract_{contract}': 1,
        f'PaymentMethod_{payment}': 1,
        f'InternetService_{internet}': 1,
    }

    # Convert to DataFrame and fill missing model columns
    user_df = pd.DataFrame([input_dict])
    for col in model_columns:
        if col not in user_df.columns:
            user_df[col] = 0  # fill others with 0
    user_df = user_df[model_columns]  # ensure correct order

    # Scale and predict
    if st.button("🔍 Predict Churn"):
        try:
            input_scaled = scaler.transform(user_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ Likely to churn (Probability: {probability:.1f}%)")
            else:
                st.success(f"✅ Not likely to churn (Probability: {100 - probability:.1f}%)")

            # Show Feature Importance if available
            if hasattr(model, 'feature_importances_'):
                st.subheader("📊 Feature Importance (Top 5)")
                feat_df = pd.DataFrame({
                    'feature': model_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(5)
                fig, ax = plt.subplots()
                ax.barh(feat_df['feature'], feat_df['importance'], color='#4e79a7')
                ax.invert_yaxis()
                ax.set_xlabel("Importance")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")

# ================== 📈 INSIGHTS ==================
elif page == "📈 Insights & Graphs":
    st.title("📈 Churn Insights & Visualizations")

    st.subheader("✅ Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax.bar_label(ax.containers[0])
    st.pyplot(fig)

    st.subheader("📑 Churn by Contract Type")
    churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    fig, ax = plt.subplots()
    ax.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
    ax.bar_label(ax.containers[0], fmt='%.1f%%')
    ax.set_ylabel('Churn Rate (%)')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0) * 100
    churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
    ax.bar_label(ax.containers[0], fmt='%.1f%%')
    st.pyplot(fig)

    st.markdown("### 🧠 Key Business Insights")
    st.markdown("""
    - Month-to-month contracts show the highest churn.
    - Electronic checks are most churn-prone.
    - Short-tenure and high-monthly-charge customers are likely to churn.
    """)

# ================== 📄 RAW DATA ==================
elif page == "📄 Raw Data":
    st.title("📄 Raw Dataset")
    st.dataframe(data)
    st.caption(f"Total Records: {len(data)}")
