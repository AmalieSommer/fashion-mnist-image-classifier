from sklearn.ensemble import RandomForestClassifier
import joblib
from utils import getModelPath
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class CustomRandomForest:
    def __init__(self, n_trees=100, max_depth=50, random_state=42, num_components=50):
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=random_state
        )

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=num_components)

    
    def fit(self, X, y):
        # First scale and reduce number of features:
        X = self.scaler.fit_transform(X)
        X = self.pca.fit_transform(X)
        self.model.fit(X, y)

    def predictClass(self, X):
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return self.model.predict(X)
    
    def predictClassProb(self, X):
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return self.model.predict_proba(X)
    
    def saveModel(self, filepath='randForest_model.pkl'):
        saved_model_path = os.path.join(getModelPath(), filepath)
        bundle = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca
        }
        joblib.dump(bundle, saved_model_path)
    
    def loadModel(self, filepath='randForest_model.pkl'):
        saved_model_path = os.path.join(getModelPath(), filepath)
        saved = joblib.load(saved_model_path)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.pca = saved['pca']