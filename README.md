# Network Anomaly Detection

This project detects suspicious network traffic using UNSW-NB15 and CIC-IDS2017 data.

## Project structure

```text
Network-Anomaly-Detection/
├── app/                  # Streamlit dashboard
├── Datasets/
│   ├── UNSW-NB15/        # UNSW training and testing CSV files
│   └── CIC-IDS2017/      # CIC traffic CSV files
├── models/               # Trained model files and metrics
├── src/                  # Training code
└── requirements.txt
```

## Run the application

Open a terminal in the project folder and run:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app\streamlit_app.py
```

Open `http://localhost:8501`, choose a dataset model, and upload one or more CSV files.
The dashboard displays prediction summaries and provides downloadable prediction files.

## Train a model

UNSW-NB15:

```powershell
python src\train_model.py --dataset Datasets\UNSW-NB15\UNSW_NB15_training-set.csv --model-name unsw_nb15_model
```

CIC-IDS2017:

```powershell
python src\train_model.py --dataset <training-sample.csv> --model-name cic_ids2017_model
```

The training script saves models and metrics in `models/`.

## Output

Predictions are saved as CSV files by the dashboard. The `prediction` column contains
the model's classification for each network-flow row.

## Notes

- Use the UNSW model with UNSW-NB15 CSV files.
- Use the CIC model with CIC-IDS2017 CSV files.
- Large datasets and generated prediction files may require a CSV viewer other than Excel.
