"""Create a reproducible synthetic customer-churn dataset."""
import numpy as np
import pandas as pd

def generate_churn_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100.")

    rng = np.random.default_rng(random_state)
    tenure = rng.integers(0, 73, n_samples)
    monthly = np.round(rng.uniform(25, 145, n_samples), 2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples, p=[0.55, 0.25, 0.20]
    )
    support = rng.choice(["Yes", "No"], n_samples, p=[0.35, 0.65])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.50, 0.15])
    payment = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer"],
        n_samples, p=[0.45, 0.30, 0.25]
    )

    # Transparent synthetic rule with noise; not intended to represent real customers.
    logit = (
        1.15
        - 0.045 * tenure
        + 0.012 * (monthly - 70)
        + 1.05 * (contract == "Month-to-month")
        - 0.65 * (contract == "Two year")
        + 0.65 * (support == "No")
        + 0.35 * (internet == "Fiber optic")
        + 0.55 * (payment == "Electronic check")
        + rng.normal(0, 0.65, n_samples)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    return pd.DataFrame({
        "TenureMonths": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "TechSupport": support,
        "InternetService": internet,
        "PaymentMethod": payment,
        "Churn": churn,
    })
