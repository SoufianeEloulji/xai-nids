from zenml import pipeline
from steps.ingesting_data import ingest_data
from steps.cleaning_data import clean_data
from steps.training_model import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(data_path: str):
    """
    A ZenML pipeline that ingests, cleans, trains, and evaluates a model.

    Args:
        data_path (str): The path to the CSV file containing the data.
    """
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)