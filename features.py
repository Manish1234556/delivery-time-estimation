import numpy as np
import pandas as pd

TEXT_COLS = ["day_of_week", "zone_type", "traffic_level", "weather",
             "vehicle_type", "package_type", "package_size"]


def haversine(a, b, c, d):
    a, b, c, d = map(np.radians, [a, b, c, d])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(x))


def build_features(d):
    """Same as the notebook's build_features (three-way experiments off)."""
    d = d.copy()
    # Tier 1: route, traffic, courier, package
    d["distance_km"] = haversine(d.pickup_lat, d.pickup_lon, d.drop_lat, d.drop_lon)
    d["distance_bucket"] = pd.cut(d.distance_km, [0, 2, 5, 10, 1e3],
                                  labels=["short", "medium", "long", "very_long"]).astype(str)
    d["traffic_score"] = d.traffic_level.map({"Low": 0, "Medium": 1, "High": 2, "Jam": 3}).astype(float)
    d["stops_per_km"] = d.num_stops / d.distance_km.clip(lower=0.1)
    d["dist_plus_stops"] = d.distance_km + d.num_stops
    d["exp_band"] = pd.cut(d.courier_experience_yrs, [-1, 2, 5, 10, 100],
                           labels=["novice", "mid", "senior", "expert"]).astype(str)
    d["handling_score"] = d.package_type.map(
        {"Documents": 1, "Food": 2, "Grocery": 2, "Electronics": 3, "Fragile": 4}).astype(float) + d.is_fragile
    d["weight_x_dist"] = d.package_weight_kg * d.distance_km
    # Tier 2: weather, time, vehicle
    d["bad_weather"] = d.weather.isin(["Rainy", "Stormy", "Foggy"]).astype(float).where(d.weather.notna())
    d["weather_x_traffic"] = d.bad_weather * d.traffic_score
    d["hour"] = d.order_dt.dt.hour
    d["rush_hour"] = (d.hour.between(12, 14) | d.hour.between(19, 21)).astype(int)
    d["meal_period"] = pd.cut(d.hour, [-1, 10, 15, 18, 24],
                              labels=["morning", "lunch", "evening", "dinner"]).astype(str)
    d["vehicle_x_dist"] = d.vehicle_type.map(
        {"Bike": 1, "Scooter": 1.05, "Car": 1.2, "Cycle": 1.6}).astype(float) * d.distance_km
    # Tier 3: pairwise interactions
    d["dist_x_traffic"] = d.distance_km * d.traffic_score
    d["pkg_x_traffic"] = d.handling_score * d.traffic_score
    d["pkg_x_route"] = d.handling_score * d.distance_km
    d["dist_x_lowrating"] = d.distance_km * (5 - d.courier_rating)
    d["courier_x_traffic"] = (5 - d.courier_rating) * d.traffic_score
    d["courier_x_pkg"] = (5 - d.courier_rating) * d.handling_score
    d["zone_x_traffic"] = (d.zone_type == "Commercial").astype(float) * d.traffic_score
    return d


def make_features(raw, features):
    """Raw order row(s) -> the exact feature columns the models were trained on."""
    d = raw.copy()
    for c in TEXT_COLS:                       # same label cleaning as training
        if c in d.columns:
            d[c] = d[c].astype("string").str.strip().str.title().astype(object)
    d["order_dt"] = pd.to_datetime(d["order_date"].astype(str) + " " + d["order_time"].astype(str))
    d = build_features(d)
    return d[features]
