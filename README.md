# hospital-surge-prediction
Machine learning and SARIMA-based hospital surge prediction using HHS respiratory data

# Hospital Surge Prediction

Machine learning and time-series modeling project to predict hospital surge events using HHS respiratory data (2020–2024).

## Overview
This project compares:
- Logistic Regression
- Random Forest
- SARIMA (time-series)
- Hybrid model combining SARIMA + ML

Goal: Predict hospital surge events (>= 85% inpatient occupancy).

## Dataset
- Source: HHS Weekly Hospital Respiratory Data
- Regions: 56 (states + DC + territories)
- Time period: 2020–2024
- Features:
  - Inpatient bed occupancy
  - ICU occupancy
  - COVID, Flu, RSV hospitalizations

## Methods
- Lag-based feature engineering
- Temporal train/test split
- SARIMA modeling per region
- Random Forest classification
- Hybrid ML + SARIMA approach
- Threshold optimization (F2 score)
- Calibration analysis
- SHAP for interpretability

## Results
- Random Forest:
  - F1 = 0.853
  - PR-AUC = 0.911

- Hybrid Model (best):
  - F1 = 0.836
  - PR-AUC = 0.927

- Calibration:
  - ECE = 0.0085 (near perfect)

## Key Insight
SARIMA’s upper confidence bound improves performance more than its point forecast, capturing surge volatility.

## Tech Stack
Python, pandas, scikit-learn, statsmodels, SHAP, matplotlib

## Project Structure
- `notebooks/` → main modeling code
- `outputs/` → plots and results
- `docs/` → presentation and proposal

## Author
Ronak Singh
