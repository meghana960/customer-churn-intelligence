import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_generator import generate_churn_data
from src.model import train_model, predict_customer

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return generate_churn_data(n_samples=5000, random_state=42)

@st.cache_resource
def load_model():
    data = load_data()
    return train_model(data)

data = load_data()
model_bundle = load_model()

st.title("📊 Customer Churn Intelligence")
st.caption("An interactive machine-learning demo for exploring customer churn risk.")
st.info(
    "Demo notice: this project generates a synthetic dataset for learning and demonstration. "
    "Predictions are not validated for real business decisions."
)

tab1, tab2, tab3 = st.tabs(["Overview", "Explore data", "Predict churn"])

with tab1:
    metrics = model_bundle["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in demo data", f"{len(data):,}")
    c2.metric("Model accuracy", f"{metrics['accuracy']:.1%}")
    c3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c4.metric("Churn rate", f"{data['Churn'].mean():.1%}")

    left, right = st.columns(2)
    with left:
        fig = px.pie(data, names="Churn", title="Churn distribution",
                     color="Churn", color_discrete_sequence=["#16a085", "#e67e22"])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        churn_by_contract = data.groupby("Contract")["Churn"].mean().reset_index()
        fig = px.bar(churn_by_contract, x="Contract", y="Churn",
                     title="Churn rate by contract type", labels={"Churn": "Churn rate"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model evaluation")
    st.write("A Random Forest classifier is trained on a stratified train/test split.")
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Score"}), use_container_width=True)

with tab2:
    st.subheader("Explore the synthetic customer dataset")
    selected_contract = st.multiselect(
        "Filter by contract",
        options=sorted(data["Contract"].unique()),
        default=sorted(data["Contract"].unique())
    )
    filtered = data[data["Contract"].isin(selected_contract)]
    st.write(f"Showing {len(filtered):,} of {len(data):,} records.")
    st.dataframe(filtered.head(100), use_container_width=True)
    fig = px.histogram(filtered, x="TenureMonths", color="Churn", nbins=24,
                       title="Customer tenure distribution by churn")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Estimate churn risk for a sample customer")
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 18)
        monthly = st.slider("Monthly charges", 20.0, 150.0, 75.0, 1.0)
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        support = st.selectbox("Tech support", ["No", "Yes"])
    with col3:
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Credit card", "Bank transfer"])

    if st.button("Predict churn risk", type="primary"):
        customer = {
            "TenureMonths": tenure,
            "MonthlyCharges": monthly,
            "Contract": contract,
            "TechSupport": support,
            "InternetService": internet,
            "PaymentMethod": payment,
        }
        probability = predict_customer(model_bundle["pipeline"], customer)
        st.metric("Estimated churn probability", f"{probability:.1%}")
        if probability >= 0.60:
            st.warning("Higher estimated churn risk in this demo model.")
        elif probability >= 0.35:
            st.info("Moderate estimated churn risk in this demo model.")
        else:
            st.success("Lower estimated churn risk in this demo model.")
        st.caption("This is a synthetic educational model, not a real customer assessment.")
