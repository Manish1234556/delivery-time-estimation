# 🚚 Delivery Time Predictor

Predicts the delivery time (in minutes) of an order from route, traffic, weather,
courier and package details. The project compares traditional machine learning
with deep learning and serves the models through a Streamlit app.

**Live app:** https://delivery-time-estimation-catalyst-crew.streamlit.app

## Overview
- Data: 6,000 delivery orders (instant-delivery type, target in minutes)
- Task: regression (predict delivery time)
- Two tracks: **Machine Learning** (Linear Regression, Random Forest, XGBoost)
  and **Deep Learning** (ANN)
- App options: Machine Learning, Deep Learning, or the automatically selected best model

## Workflow
1. **Cleaning:** consistent labels, duplicate and impossible-coordinate removal,
   extreme target values removed, missing values imputed (median / most frequent)
2. **Feature engineering:** haversine distance, distance bucket, ordered traffic score,
   courier experience band, package handling score, rush-hour and meal-period flags,
   weather x traffic, and pairwise interactions (distance x traffic, package x route,
   courier x distance, ...)
3. **Feature selection:** correlation filter (pairs above 0.9 removed)
4. **Split:** time-based 70/15/15 (train / validation / test), so the models are tested
   on later orders than they trained on
5. **Modelling:** log-transformed target, preprocessing fitted on the training set only,
   hyperparameter search with time-ordered cross-validation
6. **ANN:** 128-64-32 dense layers with batch normalization and dropout, Huber loss,
   early stopping and learning-rate scheduling

## Results

Held-out test set (later orders):

| Model | RMSE (min) | MAE (min) | R² |
|---|---|---|---|
| **XGBoost** | **3.56** | **2.83** | **0.939** |
| ANN | 3.70 | 2.85 | 0.934 |
| Linear Regression | 4.02 | 3.04 | 0.923 |
| Random Forest | 4.56 | 3.60 | 0.900 |

Cross-validation on the training data (5 time-ordered folds, mean +/- std):

| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| XGBoost | 4.26 +/- 0.53 | 3.19 +/- 0.21 | 0.918 +/- 0.022 |
| Linear Regression | 4.41 +/- 0.47 | 3.23 +/- 0.18 | 0.912 +/- 0.021 |
| ANN | 4.83 +/- 1.09 | 3.58 +/- 0.57 | 0.890 +/- 0.056 |
| Random Forest | 5.57 +/- 0.62 | 4.16 +/- 0.26 | 0.860 +/- 0.033 |

XGBoost is the best model: it has the lowest error on both the test set and in
cross-validation. Its typical error is about 2.8 minutes (around 7% of the average
delivery time).

## Using the app
1. Choose a model in the sidebar: **Machine Learning**, **Deep Learning** or **Best model**
2. Fill in the order, route, courier and package details
3. Press **Predict delivery time** to see the estimate, the likely range, the expected
   arrival time and the predictions of both models side by side
4. The **Model comparison** tab shows the test results of all models

## Run the app locally
```
git clone https://github.com/Manish1234556/delivery-time-estimation.git
cd delivery-time-estimation
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501

## Run with Docker
```
git clone https://github.com/Manish1234556/delivery-time-estimation.git
cd delivery-time-estimation
docker build -t delivery-time-predictor .
docker run -p 8501:8501 delivery-time-predictor
```
Then open http://localhost:8501

The first build takes several minutes and the image is large (around 1.5 to 2 GB)
because it includes TensorFlow.

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `features.py` | Feature engineering shared by training and the app |
| `meta.json` | Feature list, input options, defaults and test metrics |
| `ml_model.pkl` | Trained XGBoost pipeline (preprocessing + model) |
| `ann_model.keras` | Trained ANN |
| `ann_preprocessor.pkl` | Preprocessing pipeline for the ANN |
| `requirements.txt` | Python dependencies (pinned versions) |
| `Dockerfile`, `.dockerignore` | Container setup |
| `code_part.ipynb` | Full notebook: cleaning, features, training, evaluation |
| `delivery_time_dataset.csv` | Dataset |

## Limitations
- The extreme-target fence (Q3 + 3xIQR) was computed on the full dataset before the
  train/validation/test split. This is a mild form of leakage and removes some hard
  cases from the test set, so test scores may be slightly optimistic.
- Linear Regression is close to XGBoost, which suggests the relationships in this data
  are largely linear; the gain from complex models is modest.
- Results come from a single dataset and may not transfer to other cities or
  delivery types without retraining.
- Pickled models depend on the library versions in `requirements.txt`; use those
  versions to avoid loading errors.
