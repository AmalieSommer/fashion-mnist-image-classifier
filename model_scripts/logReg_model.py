from sklearn.linear_model import LogisticRegression
import joblib
from utils import getModelPath
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CustomLogisticRegression:
    def __init__(self, max_iter=10000, random_state=42, pca_components=50):
        self.model = LogisticRegression(
            multi_class='multinomial',
            random_state=random_state,
            max_iter=max_iter
        )

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)


    def fit(self, X, y):
        X = self.scaler.fit_transform(X) #Fits the scaler and PCA to the X_training data
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
    

    # Saves the fitted model, scaler and pca components:
    def saveModel(self, filepath='logreg_model.pkl'):
        saved_model_path = os.path.join(getModelPath(), filepath)

        bundle = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca
        }

        joblib.dump(bundle, saved_model_path)
        
    
    def loadModel(self, filepath='logreg_model.pkl'):
        saved_model_path = os.path.join(getModelPath(), filepath)
        saved = joblib.load(saved_model_path)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.pca = saved['pca']

        
        
    