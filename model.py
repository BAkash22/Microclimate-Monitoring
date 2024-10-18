import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


data = pd.read_csv('chengalpattu_crops_dataset.csv')


X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['crop']


model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X, y)


with open('crop_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    
    with open('crop_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
  
    crop_prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    
    return crop_prediction[0]


