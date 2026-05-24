import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SentimentTrainer:

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.sentiment_data_path = self.root_dir / "data" / "databases" / "sentiment.csv"
        if Path("/data").exists():
            self.model_dir = Path("/data/models")
        else:
            self.model_dir = self.root_dir / "data" / "datasets" / "models"
        self.model_path = self.model_dir / "sentiment_pipeline.pkl"
        self.encoder_path = self.model_dir / "sentiment_label_encoder.pkl"

    def load_data(self):
        if not self.sentiment_data_path.exists():
            raise FileNotFoundError(f"Sentiment data not found at: {self.sentiment_data_path}")
        df = pd.read_csv(self.sentiment_data_path)
        df.columns = ['text', 'sentiment']
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['sentiment'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(label_encoder, self.encoder_path)
        return df, label_encoder

    def train(self):
        df, _ = self.load_data()
        X = df['text']
        y = df['label']
        logger.info("Starting sentiment model training...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=None)),
        ], memory=None)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)
        logger.info("Model trained. Test Accuracy: %.4f", accuracy)
        joblib.dump(pipeline, self.model_path)
        logger.info("Sentiment model pipeline saved to %s", self.model_path)
        return {"status": "success", "accuracy": accuracy, "model_path": str(self.model_path)}

    def __call__(self):
        return self.train()
        
trainer = SentimentTrainer()
