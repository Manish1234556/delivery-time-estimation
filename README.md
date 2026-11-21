# Delivery Time Predictor

Predicts delivery time (in minutes) for an order from route, traffic, weather,
courier and package details.

## Models
- **Machine Learning:** XGBoost (best model)
- **Deep Learning:** ANN (128-64-32, batch norm, dropout)
- Baselines: Linear Regression, Random Forest

## Results (held-out, later-in-time test set)
| Model | RMSE (min) | MAE (min) | R² |
|---|---|---|---|
| XGBoost | 3.56 | 2.83 | 0.939 |
| ANN | 3.70 | 2.85 | 0.934 |
| Linear Regression | 4.02 | 3.04 | 0.923 |
| Random Forest | 4.56 | 3.60 | 0.900 |

## App
The Streamlit app lets you choose Machine Learning, Deep Learning, or the
automatically selected best model.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

## Files
- `app.py`: Streamlit UI
- `features.py`: feature engineering shared by training and the app
- `meta.json`, `ml_model.pkl`, `ann_model.keras`, `ann_preprocessor.pkl`: trained models and metadata
