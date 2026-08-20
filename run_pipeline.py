from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":

    data_path = "data/dataset_complet.csv"  
    train_pipeline(data_path=data_path)