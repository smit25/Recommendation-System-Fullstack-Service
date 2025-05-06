from model.matrix_factorization import MatrixFactorization
from utils.clean_data import clean_data
from utils.sql import get_ratings_df

class ModelTrainer:
    def __init__(self, version: str, pipeline_version: str):
        self.model = MatrixFactorization(version, pipeline_version)

    def train(self):
        df = get_ratings_df()
        (X_train, _, _), _, _ = clean_data(df)
        self.model.train(X_train)
        self.model.save_model()
