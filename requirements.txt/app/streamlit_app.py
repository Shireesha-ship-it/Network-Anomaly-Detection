from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "UNSW-NB15": ROOT / "Datasets" / "UNSW-NB15",
    "CIC-IDS2017": ROOT / "Datasets" / "CIC-IDS2017",
}
MODELS = {
    "UNSW-NB15": ROOT / "models" / "unsw_nb15_model.joblib",
    "CIC-IDS2017": ROOT / "models" / "cic_ids2017_model.joblib",
}

st.set_page_config(page_title="Network Anomaly Detection", page_icon="🛡️")
st.title("Network Anomaly Detection")
dataset_name = st.selectbox("Choose dataset model", list(DATASETS))
st.write("Upload a CSV containing network-flow columns to classify each row.")

model_path = MODELS[dataset_name]
if not model_path.exists():
    st.warning(f"No {dataset_name} model found yet. Train it before uploading data.")
    st.stop()

uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
if uploaded_files:
    prediction_frames = []
    for uploaded in uploaded_files:
        data = pd.read_csv(uploaded)
        st.write(f"Processing **{uploaded.name}** ({len(data):,} rows)")
        if "Label" in data.columns:
            detected_dataset = "CIC-IDS2017"
            feature_data = data.drop(columns=["Label"])
        elif "label" in data.columns and "dur" in data.columns:
            detected_dataset = "UNSW-NB15"
            feature_data = data.drop(columns=["label", "attack_cat"], errors="ignore")
        else:
            st.error(f"{uploaded.name} is not recognized as UNSW-NB15 or CIC-IDS2017.")
            continue

        if detected_dataset != dataset_name:
            st.info(f"Detected {detected_dataset} columns, so the {detected_dataset} model will be used.")
        selected_model_path = MODELS[detected_dataset]
        model = joblib.load(selected_model_path)
        expected_columns = set(model.named_steps["preprocessor"].feature_names_in_)
        missing_columns = expected_columns.difference(feature_data.columns)
        if missing_columns:
            st.error(
                f"{uploaded.name} is missing {len(missing_columns)} columns required by "
                f"the {detected_dataset} model."
            )
            continue

        result = data.copy()
        result["prediction"] = model.predict(feature_data)
        prediction_frames.append(result)
        st.dataframe(result.head(100), use_container_width=True)
        st.download_button(
            f"Download {uploaded.name} predictions",
            result.to_csv(index=False).encode("utf-8"),
            f"{Path(uploaded.name).stem}_predictions.csv",
            "text/csv",
            key=f"download-{uploaded.name}",
        )

    if prediction_frames:
        combined_result = pd.concat(prediction_frames, ignore_index=True)
        prediction_counts = combined_result["prediction"].value_counts().rename_axis("prediction").to_frame("rows")
        total_rows = len(combined_result)
        if "Benign" in prediction_counts.index:
            suspicious_rows = total_rows - int(prediction_counts.loc["Benign", "rows"])
        elif 0 in prediction_counts.index or "0" in prediction_counts.index:
            normal_key = 0 if 0 in prediction_counts.index else "0"
            suspicious_rows = total_rows - int(prediction_counts.loc[normal_key, "rows"])
        else:
            suspicious_rows = total_rows
        metric_columns = st.columns(3)
        metric_columns[0].metric("Rows processed", f"{total_rows:,}")
        metric_columns[1].metric("Prediction classes", len(prediction_counts))
        metric_columns[2].metric("Potential anomalies", f"{suspicious_rows:,}")

        st.subheader("Prediction summary")
        st.bar_chart(prediction_counts)
        if "Label" in combined_result.columns:
            accuracy = accuracy_score(
                combined_result["Label"].astype(str),
                combined_result["prediction"].astype(str),
            )
            st.metric("Accuracy against provided labels", f"{accuracy:.2%}")

        st.download_button(
            "Download all predictions",
            combined_result.to_csv(index=False).encode("utf-8"),
            "cic_ids2017_predictions.csv",
            "text/csv",
        )
    st.stop()
