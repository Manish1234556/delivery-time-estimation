import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import make_features, haversine

BASE = Path(__file__).parent
st.set_page_config(page_title="Delivery Time Predictor", page_icon="🚚", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
div[data-testid="stMetricValue"] {font-size: 2.6rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_all():
    from tensorflow.keras.models import load_model
    meta = json.load(open(BASE / "meta.json"))
    ml = joblib.load(BASE / "ml_model.pkl")
    ann = load_model(BASE / "ann_model.keras")
    ann_prep = joblib.load(BASE / "ann_preprocessor.pkl")
    return meta, ml, ann, ann_prep


meta, ml_model, ann_model, ann_prep = load_all()
opts, dfl = meta["options"], meta["defaults"]
ML_NAME, BEST = meta["ml_name"], meta["best_overall_name"]


def predict_ml(X):
    return float(ml_model.predict(X)[0])


def predict_dl(X):
    A = ann_prep.transform(X)
    A = A.toarray() if hasattr(A, "toarray") else A
    return float(np.expm1(ann_model.predict(np.asarray(A, dtype=np.float32), verbose=0).flatten()[0]))


def metric_row(name):
    return next((m for m in meta["metrics"] if m["Model"] == name), None)


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Model")
    ml_label, dl_label = f"Machine Learning ({ML_NAME})", "Deep Learning (ANN)"
    best_label = f"⭐ Best model ({BEST})"
    choice = st.radio("Choose a model", [ml_label, dl_label, best_label], label_visibility="collapsed")
    st.caption({ml_label: "Gradient-boosted trees: strong on tabular data, fast.",
                dl_label: "Neural network (128-64-32) trained on the same features.",
                best_label: "Automatically uses the model with the lowest test RMSE."}[choice])
    st.divider()
    r = metric_row(ML_NAME if choice == ml_label else "ANN" if choice == dl_label else BEST)
    if r:
        st.markdown("**Test performance**")
        st.metric("Typical error (MAE)", f"{r['Test MAE']:.2f} min")
        st.metric("RMSE", f"{r['Test RMSE']:.2f} min")
        st.metric("R²", f"{r['Test R2']:.3f}")

st.title("🚚 Delivery Time Predictor")
tab_pred, tab_cmp = st.tabs(["🔮 Predict", "📊 Model comparison"])

# ---------------- predict tab ----------------
with tab_pred:
    with st.form("order"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("##### 📦 Order")
            order_date = st.date_input("Order date")
            order_time = st.time_input("Order time", value=pd.to_datetime("19:00").time())
            zone_type = st.selectbox("Zone type", opts["zone_type"], index=opts["zone_type"].index(dfl["zone_type"]))
            num_stops = st.number_input("Number of stops", 0, 20, int(round(dfl["num_stops"])))
            holiday = st.checkbox("Public holiday")
            festive = st.checkbox("Festive season")
        with b:
            st.markdown("##### 🚦 Conditions & route")
            traffic = st.select_slider("Traffic level", opts["traffic_level"]
                                       if set(opts["traffic_level"]) != {"Low", "Medium", "High", "Jam"}
                                       else ["Low", "Medium", "High", "Jam"], value=dfl["traffic_level"])
            weather = st.selectbox("Weather", opts["weather"], index=opts["weather"].index(dfl["weather"]))
            p1, p2 = st.columns(2)
            pickup_lat = p1.number_input("Pickup lat", value=float(dfl["pickup_lat"]), format="%.5f")
            pickup_lon = p2.number_input("Pickup lon", value=float(dfl["pickup_lon"]), format="%.5f")
            d1, d2 = st.columns(2)
            drop_lat = d1.number_input("Drop lat", value=float(dfl["drop_lat"]), format="%.5f")
            drop_lon = d2.number_input("Drop lon", value=float(dfl["drop_lon"]), format="%.5f")
        with c:
            st.markdown("##### 🏍️ Courier & package")
            vehicle = st.selectbox("Vehicle", opts["vehicle_type"], index=opts["vehicle_type"].index(dfl["vehicle_type"]))
            k1, k2 = st.columns(2)
            c_age = k1.number_input("Age", 18, 70, int(round(dfl["courier_age"])))
            c_rating = k2.number_input("Rating", 1.0, 5.0, float(round(dfl["courier_rating"], 1)), step=0.1)
            k3, k4 = st.columns(2)
            c_exp = k3.number_input("Experience (yrs)", 0.0, 40.0, float(round(dfl["courier_experience_yrs"], 1)), step=0.5)
            c_deliv = k4.number_input("Deliveries done", 0, 100000, int(round(dfl["courier_deliveries_completed"])))
            q1, q2 = st.columns(2)
            p_type = q1.selectbox("Package type", opts["package_type"], index=opts["package_type"].index(dfl["package_type"]))
            p_size = q2.selectbox("Size", opts["package_size"], index=opts["package_size"].index(dfl["package_size"]))
            q3, q4 = st.columns(2)
            p_weight = q3.number_input("Weight (kg)", 0.1, 100.0, float(round(dfl["package_weight_kg"], 1)), step=0.1)
            prep_min = q4.number_input("Prep time (min)", 0.0, 120.0, float(round(dfl["preparation_time_min"], 1)), step=1.0)
            fragile = st.checkbox("Fragile")
        go = st.form_submit_button("🔮 Predict delivery time", type="primary", use_container_width=True)

    if go:
        raw = pd.DataFrame([{
            "order_date": order_date.strftime("%Y-%m-%d"), "order_time": order_time.strftime("%H:%M"),
            "day_of_week": order_date.strftime("%A"), "is_weekend": int(order_date.weekday() >= 5),
            "is_public_holiday": int(holiday), "is_festive_season": int(festive),
            "pickup_lat": pickup_lat, "pickup_lon": pickup_lon, "drop_lat": drop_lat, "drop_lon": drop_lon,
            "zone_type": zone_type, "num_stops": num_stops, "traffic_level": traffic, "weather": weather,
            "courier_age": c_age, "courier_rating": c_rating, "courier_experience_yrs": c_exp,
            "courier_deliveries_completed": c_deliv, "vehicle_type": vehicle, "package_type": p_type,
            "package_size": p_size, "package_weight_kg": p_weight, "is_fragile": int(fragile),
            "preparation_time_min": prep_min}])
        X = make_features(raw, meta["features"])

        p_ml, p_dl = predict_ml(X), predict_dl(X)
        if choice == ml_label:
            used, pred = ML_NAME, p_ml
        elif choice == dl_label:
            used, pred = "ANN", p_dl
        else:
            used, pred = BEST, (p_dl if meta["best_overall"] == "DL" else p_ml)

        mae = (metric_row(used) or {}).get("Test MAE", 3.0)
        eta = datetime.combine(order_date, order_time) + timedelta(minutes=pred)
        dist = float(haversine(pickup_lat, pickup_lon, drop_lat, drop_lon))

        st.divider()
        st.success(f"Prediction from **{used}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated delivery time", f"{pred:.1f} min")
        m2.metric("Likely range", f"{max(pred - mae, 0):.0f} to {pred + mae:.0f} min")
        m3.metric("Expected arrival", eta.strftime("%H:%M"))
        m4.metric("Route distance", f"{dist:.1f} km")

        st.markdown("##### All models on this order")
        st.bar_chart(pd.DataFrame({"Predicted minutes": [p_ml, p_dl]}, index=[ML_NAME, "ANN"]),
                     horizontal=True, height=160)
        if abs(p_ml - p_dl) > 8:
            st.warning("The two models disagree by more than 8 minutes. This order may be unusual "
                       "(check coordinates and inputs).")

# ---------------- comparison tab ----------------
with tab_cmp:
    comp = pd.DataFrame(meta["metrics"])
    comp.insert(0, "Pick", np.where(comp["Model"] == BEST, "✅ best", ""))
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.bar_chart(comp.set_index("Model")["Test RMSE"], height=260)
    st.caption("RMSE and MAE are in minutes (lower is better). R² closer to 1 is better. "
               "All models were evaluated on the same later-in-time test set.")
