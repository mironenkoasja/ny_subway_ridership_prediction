# Subway Turnstile Forecasting Pipeline

This project builds an end-to-end **batch machine learning pipeline** for predicting NYC subway ridership based on historical turnstile data. The project runs locally using Docker Compose.

🚇 **What does it do?**  
The system forecasts **ridership volume per station group in 4-hour windows**, using patterns from the previous days and weeks. 

**Tools used:**

- **PostgreSQL** to store preprocessed features
- **XGBoost** as the core model
- **Hyperopt** for hyperparameter tuning
- **MLflow** for experiment tracking 
- **Apache Airflow** to orchestrate the pipeline
- **Docker** for reproducible deployment
-**Gafana** for monitoring prediction quality

Although the model is trained on historical data from **2022**, the Airflow DAG is designed to **simulate daily predictions in real time** by executing a rolling prediction window as if today were the current date.

After each prediction run, the DAG also generates **monitoring data** to help evaluate model performance and track changes over time.

## 📦 Project Structure

```bash
.
├── dags/
│   └── turnstile_ml_pipeline.py       # Main pipeline DAG      
├── pipeline/ 
│   ├── download_ingest.py             # Download data for training
│   ├── preprocess.py                  # Preprocessing and feature engineering (training and inference)         
│   ├── tune_model.py                  # Hyperparameter tuning with Hyperopt
│   ├── train_register.py              # Model training and MLflow experiment tracking
│   ├── inference.py                   # Prediction
│   └── monitor.py                     # Save data for monitoring
├── notebooks/                         # Jupyter notebooks for exploration
├── data/                              # Raw and processed data 
├── db_int/                          
├── grafana_storage/                  # Persistent volume for Grafana
├── docker-compose.yml                # Docker setup for MLflow, Airflow, Postgres, etc.
├── Dockerfile.airflow                   
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview and instructions

```

## 🛠 Features

- Forecasts **ridership volume per turnstile group** for 4-hour intervals
- Uses historical patterns including previous day/hour/week statistics
- Automatically selects best hyperparameters with **Hyperopt**
- Trains and save best model with **MLflow**
- Supports **daily prediction simulation** using Airflow scheduler
- Saves prediction metrics and monitoring artifacts for further analysis

## 🏃 Usage

1. **Start the system:**

   ```bash
   docker-compose up --build -d
   ```

2. **Access the services:**

   - Airflow: [http://localhost:8080](http://localhost:8080)
   - MLflow UI: [http://localhost:5000](http://localhost:5000)
   - Postgres: localhost:5432

3. **Trigger the pipeline manually or via scheduler** in Airflow

4. **Inspect predictions and metrics** in Grafana and the monitoring artifacts

## 🔍 Use Cases

This forecasting pipeline helps estimate subway **turnstile congestion** by location and time. It can be used by:

- **Transit authorities** to allocate station personnel more efficiently
- **Data teams** to monitor ridership trends
- **Researchers** interested in urban mobility forecasting

## 📈 Monitoring Output

After predictions, the DAG generates:

- Metrics (RMSE, MAE)
- MAE vs Mean ridership
- Avarage MAE for Group of turstiles of different load

These artifacts are logged and can be used to evaluate model degradation over time.

 ## 🧪 Unit tests & Code Quality

This project includes unit tests covering the key pipeline components, including data ingestion, preprocessing, model training, tuning, inference, and monitoring.
Tests are automatically executed when all containers are started.

In addition to testing, the project automatically runs:

Code formatting checks using black

Linting checks using flake8

These ensure the code is consistent, readable, and adheres to Python best practices.
Test results and logs are saved to: 

   ```bash
   tests/tests_logs.txt
   tests/format_log.txt
   tests/lint_log.txt
   ```



