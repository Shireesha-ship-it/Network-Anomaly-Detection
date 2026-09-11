# Network Anomaly Detection Project Report

## 1. Objective

The project identifies normal and suspicious network traffic from labeled network-flow CSV files. It provides a reusable machine-learning pipeline and a Streamlit dashboard for uploading files and downloading predictions.

## 2. Datasets

- **UNSW-NB15:** training and testing traffic CSV files.
- **CIC-IDS2017:** eight traffic CSV files covering benign traffic and multiple attack categories.

The datasets are stored in `Datasets/UNSW-NB15` and `Datasets/CIC-IDS2017`.

## 3. Method

The training pipeline:

1. Loads a labeled CSV file.
2. Removes label-only columns from the feature set.
3. Replaces infinite values and imputes missing values.
4. One-hot encodes categorical features.
5. Standardizes numeric features.
6. Trains a class-balanced Logistic Regression classifier.
7. Saves the model and classification metrics in `models/`.

Separate models are used because UNSW-NB15 and CIC-IDS2017 have different feature schemas.

## 4. Application

The Streamlit dashboard supports:

- UNSW-NB15 and CIC-IDS2017 model selection
- Automatic CSV format detection
- Uploading one or multiple CSV files
- Prediction summaries and charts
- Accuracy display when labels are present
- Individual and combined prediction downloads

Start the application with:

```powershell
streamlit run app\streamlit_app.py
```

Then open `http://localhost:8501`.

## 5. Results

- UNSW-NB15 smoke test: passed with 25 test rows.
- CIC-IDS2017 smoke test: passed with 25 test rows.
- Streamlit dashboard: responded successfully with HTTP 200.
- Combined CIC-IDS2017 prediction output: 2,313,810 rows.

## 6. Output

Prediction CSV files contain the original input columns plus a `prediction` column. The final combined CIC-IDS2017 output was saved as:

```text
C:\Users\bhara\Downloads\cic_ids2017_all_predictions.csv
```

## 7. Conclusion

The project provides an end-to-end workflow from network-flow datasets to trained anomaly classifiers, interactive predictions, visualization, and downloadable results.
