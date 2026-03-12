import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load model and training data
# ----------------------------
st.set_page_config(page_title="Forest Cover Type Predictor", layout="wide")

model = joblib.load("artifacts/xgb_best_model.joblib")
X_train = pd.read_parquet("artifacts/X_train.parquet")

# Separate columns
base_cols = ['Elevation','Aspect','Slope',
             'Horizontal_Distance_To_Hydrology',
             'Vertical_Distance_To_Hydrology',
             'Horizontal_Distance_To_Roadways',
             'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
             'Horizontal_Distance_To_Fire_Points']
wilderness_cols = [c for c in X_train.columns if c.startswith("Wilderness_Area")]
soil_cols       = [c for c in X_train.columns if c.startswith("Soil_Type")]

# ----------------------------
# Styling header
# ----------------------------
st.markdown(
    "<h1 style='text-align:center;'>🌲 Forest Cover Type Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)
st.write("Interactively predict forest cover type (1–7) using environmental features "
         "or upload a CSV for batch predictions.")

# ----------------------------
# Helper: Auto slider
# ----------------------------
def auto_slider(col, step=1):
    mn, mx = int(X_train[col].min()), int(X_train[col].max())
    default = int(X_train[col].median())
    return st.slider(f"{col}", mn, mx, default, step=step)

# ----------------------------
# Single prediction inputs
# ----------------------------
st.subheader("🔹 Single Prediction")
col1, col2 = st.columns(2)

with col1:
    elev = auto_slider("Elevation", step=1)
    aspect = auto_slider("Aspect", step=1)
    slope = auto_slider("Slope", step=1)
    hd_hydro = auto_slider("Horizontal_Distance_To_Hydrology", step=1)
    vd_hydro = auto_slider("Vertical_Distance_To_Hydrology", step=1)
    hd_road  = auto_slider("Horizontal_Distance_To_Roadways", step=1)

with col2:
    h9  = auto_slider("Hillshade_9am", step=1)
    hn  = auto_slider("Hillshade_Noon", step=1)
    h3  = auto_slider("Hillshade_3pm", step=1)
    hd_fire = auto_slider("Horizontal_Distance_To_Fire_Points", step=1)

wilderness = st.selectbox("Choose Wilderness Area", wilderness_cols)
soil_type  = st.selectbox("Choose Soil Type", list(range(1, 41)))

# Build feature row
single_row = {c:0 for c in X_train.columns}
for k,v in zip(
    ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
     'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
     'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points'],
    [elev,aspect,slope,hd_hydro,vd_hydro,hd_road,h9,hn,h3,hd_fire]
): single_row[k] = v
single_row[wilderness] = 1
single_row[f"Soil_Type{soil_type}"] = 1
single_df = pd.DataFrame([single_row])

if st.button("🔍 Predict Cover Type"):
    proba = model.predict_proba(single_df)[0]
    pred  = np.argmax(proba) + 1
    st.success(f"**Predicted Cover Type:** {pred}")
    st.bar_chart(pd.Series(proba, index=[f"Type {i}" for i in range(1,8)]))

st.markdown("---")

# ----------------------------
# Batch Prediction
# ----------------------------
st.subheader("📂 Batch Prediction from CSV")
st.caption("Upload a CSV with **exactly the same column structure** as training data (no Cover_Type).")
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    if set(X_train.columns).issubset(df.columns):
        preds = model.predict(df[X_train.columns]) + 1
        df_out = df.copy()
        df_out["Predicted_Cover_Type"] = preds
        st.write(df_out.head())
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Predictions", data=csv,
                           file_name="cover_type_predictions.csv",
                           mime="text/csv")
    else:
        st.error("Uploaded CSV does not match required columns.")
