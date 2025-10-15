from sklearn.ensemble import RandomForestClassifier
import joblib
from utils import getModelPath
import os


class CustomRandomForest:
    def __init__(self, n_trees=100, max_depth=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=random_state
        )

    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predictClass(self, X):
        return self.model.predict(X)
    
    def predictClassProb(self, X):
        return self.model.predict_proba(X)
    
    def saveModel(self, filepath='randForest_model.pkl'):
        saved_model_path = os.path.join(getModelPath(), filepath)
        joblib.dump(self.model, saved_model_path)
    
    def loadModel(self, filepath='randForest_model.pkl'):
        saved_model_path = os.path.join(getModelPath(), filepath)
        self.model = joblib.load(saved_model_path)