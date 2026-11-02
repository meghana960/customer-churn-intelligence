from src.data_generator import generate_churn_data
from src.model import train_model, predict_customer

def test_data_generation_shape_and_target():
    data = generate_churn_data(200, 7)
    assert len(data) == 200
    assert set(data["Churn"].unique()).issubset({0, 1})

def test_training_and_prediction():
    data = generate_churn_data(500, 7)
    bundle = train_model(data)
    customer = {
        "TenureMonths": 12,
        "MonthlyCharges": 80.0,
        "Contract": "Month-to-month",
        "TechSupport": "No",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    probability = predict_customer(bundle["pipeline"], customer)
    assert 0.0 <= probability <= 1.0
