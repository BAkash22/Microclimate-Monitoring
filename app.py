import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the historical crop data
historical_data = pd.read_csv('historical_crops_data.csv')  # Ensure this CSV is structured as Year,Month,Crop,Season

# Load the trained model
model = pickle.load(open('crop_prediction_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.get_json()
    N = float(data['N'])
    P = float(data['P'])
    K = float(data['K'])
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    ph = float(data['ph'])
    rainfall = float(data['rainfall'])
    season = data['season']  # Get the season from input
    year = 2023  # Assuming current year for historical comparison

    # Make prediction using the loaded model
    final_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(final_features)
    predicted_crop = prediction[0]

    # Load the historical crop data for the specified season and year
    historical_crops = historical_data[(historical_data['Season'] == season) & (historical_data['Year'] == year)]

    # Check if there are any crops grown in the specified season and year
    if historical_crops.empty:
        historically_grown_text = f"No crops historically grown in {season} of year {year}."
        historically_grown = []
    else:
        # Get unique crops grown in the specified season
        historically_grown = historical_crops['Crop'].unique()
        historically_grown_text = f'Crops historically grown in {season}: {", ".join(historically_grown)}'

    if predicted_crop in historically_grown:
        comparison_result = f"The predicted crop '{predicted_crop}' has been historically grown in this period."
    else:
        comparison_result = f"The predicted crop '{predicted_crop}' has NOT been historically grown in this period."

    return jsonify({
        'predicted_crop': predicted_crop,
        'comparison_text': comparison_result,
        'historical_grown_text': historically_grown_text
    })

if __name__ == '__main__':
    app.run(debug=True)
