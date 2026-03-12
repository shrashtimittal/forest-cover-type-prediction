import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Forest Cover Type Predictor", layout="wide")
st.title("🌲 Forest Cover Type Predictor")

# Load trained model and column order
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/xgb_best_model.joblib")
    X_train = pd.read_parquet("artifacts/X_train.parquet")
    return model, list(X_train.columns)

model, feature_cols = load_model()
st.write("Model loaded. Provide input values below to predict cover type (1–7).")

# ---- Input Widgets ----
col1, col2 = st.columns(2)

# Key numeric features for quick manual entry
elevation = col1.number_input("Elevation", min_value=1800, max_value=4000, value=2500)
aspect    = col1.slider("Aspect (0-360)", 0, 360, 180)
slope     = col1.slider("Slope (degrees)", 0, 60, 10)
hd_hydro  = col1.number_input("Horizontal_Distance_To_Hydrology", 0, 5000, 100)
vd_hydro  = col1.number_input("Vertical_Distance_To_Hydrology", -500, 500, 0)
hd_road   = col1.number_input("Horizontal_Distance_To_Roadways", 0, 7000, 500)
hill9     = col2.slider("Hillshade_9am", 0, 255, 200)
hillnoon  = col2.slider("Hillshade_Noon", 0, 255, 220)
hill3     = col2.slider("Hillshade_3pm", 0, 255, 200)
hd_fire   = col2.number_input("Horizontal_Distance_To_Fire_Points", 0, 7000, 500)

# Wilderness Area (one-hot)
st.subheader("Wilderness Area")
wild_area = st.selectbox(
    "Choose Wilderness Area",
    ["Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4"]
)

# Soil Type (one-hot)
st.subheader("Soil Type")
soil_type = st.selectbox("Choose Soil Type (1–40)", list(range(1,41)))
soil_col  = f"Soil_Type{soil_type}"

# ---- Assemble Input Vector ----
def build_input():
    data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
    data["Elevation"] = elevation
    data["Aspect"] = aspect
    data["Slope"] = slope
    data["Horizontal_Distance_To_Hydrology"] = hd_hydro
    data["Vertical_Distance_To_Hydrology"] = vd_hydro
    data["Horizontal_Distance_To_Roadways"] = hd_road
    data["Hillshade_9am"] = hill9
    data["Hillshade_Noon"] = hillnoon
    data["Hillshade_3pm"] = hill3
    data["Horizontal_Distance_To_Fire_Points"] = hd_fire
    # one-hot selections
    if wild_area in data.columns:
        data[wild_area] = 1
    if soil_col in data.columns:
        data[soil_col] = 1
    return data

if st.button("🔍 Predict Cover Type"):
    X_new = build_input()
    pred_class = model.predict(X_new)[0] + 1   # shift back to 1–7
    st.success(f"**Predicted Forest Cover Type: {int(pred_class)}**")
