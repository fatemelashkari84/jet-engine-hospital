# Jet Engine Hospital

**Predicting Failure Before It Happens**

An early-warning dashboard for turbofan engine health monitoring, built on the NASA C-MAPSS
(Commercial Modular Aero-Propulsion System Simulation) dataset. The system combines Remaining
Useful Life (RUL) regression, failure-horizon classification, and unsupervised anomaly detection
into a single, auditable maintenance recommendation.

## What it does

For a selected engine and cycle, the dashboard reports:

- **RUL estimate** — predicted Remaining Useful Life (in cycles), with a prediction interval
- **Failure risk** — calibrated probability of failure within the next 10, 20, and 30 cycles
- **Anomaly score** — an Isolation Forest–based abnormality score, independent of failure labels
- **Recommendation** — a `CONTINUE` / `INSPECT` / `STOP` action, with the specific signals that triggered it

## Live app

The dashboard is deployed on Streamlit Community Cloud: *https://jet-engine-hospital-8ae5lkuruydmgiappu9bji9.streamlit.app/*

## Repository structure

```
jet-engine-hospital/
├── app.py                     # Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── dashboard_artifacts.pkl     # Trained models, scaler, calibrators, and test data (Git LFS)
└── README.md
```

## Running locally

```bash
git clone https://github.com/fatemelashkari84/jet-engine-hospital.git
cd jet-engine-hospital
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Models used

| Task | Model |
|---|---|
| RUL regression | Random Forest Regressor |
| Failure-horizon classification (10/20/30 cycles) | Logistic Regression + isotonic calibration |
| Anomaly detection | Isolation Forest |

## Data source

NASA C-MAPSS Jet Engine Simulated Data — [NASA Open Data Portal](https://data.nasa.gov/dataset/C-MAPSS-Jet-Engine-Simulated-Data/xaut-bemq)

## Notes

- Large model artifacts (`dashboard_artifacts.pkl`) are tracked with **Git LFS**.
- This dashboard is a supervised-project deliverable; thresholds and recommendation logic are
  tuned on validation engines and should not be treated as certified maintenance guidance.
