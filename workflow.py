import pandas as pd
from prefect import task, flow, get_run_logger
# from prefect.deployments import Deployment # In case needed later, fix path
import train_model
from train_model import clean_text, train_and_log
import os
import time

@task(name="Ingest Data", retries=2, retry_delay_seconds=10)
def ingest_data(file_path):
    logger = get_run_logger()
    logger.info(f"Checking for data at {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded {len(df)} records.")
    return df

@task(name="Clean and Feature Engineer")
def preprocess_data(df):
    logger = get_run_logger()
    logger.info("Starting data cleaning process...")
    df = df.dropna(subset=['Review text', 'Ratings'])
    df['Clean_Text'] = df['Review text'].apply(clean_text)
    df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)
    logger.info("Cleaning complete.")
    return df

@task(name="Execute MLflow Training")
def run_training_pipeline():
    logger = get_run_logger()
    logger.info("Triggering MLflow training script...")
    train_and_log()
    logger.info("Training complete and logged to MLflow.")

@flow(name="Flipkart Sentiment Training Pipeline")
def training_flow():
    # File path for the dataset
    data_path = r"C:\flipkart\reviews_data_dump\reviews_badminton\data.csv"
    
    # Run tasks
    df_raw = ingest_data(data_path)
    df_clean = preprocess_data(df_raw)
    run_training_pipeline()

if __name__ == "__main__":
    # To run the flow immediately
    training_flow()
    
    # Enable automated scheduling (every day at midnight)
    training_flow.serve(
        name="flipkart-training-scheduler",
        cron="0 0 * * *", 
        description="Scheduled training for Flipkart reviews sentiment analysis"
    )
