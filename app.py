import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)


historical_data = pd.read_csv('historical_crops_data.csv') 

model = pickle.load(open('crop_prediction_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  
    data = request.get_json()
    print(f"Received data: {data}")  

    try:
       
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        season = data.get('season', 'Kharif')  # Default to 'Kharif' if not provided

          

        final_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(final_features)
        predicted_crop = prediction[0]
        print(f"Predicted crop: {predicted_crop}")  
        historical_crops = historical_data[(historical_data['Season'] == season) ]

        if historical_crops.empty:
            historically_grown_text = f"No crops historically grown in {season} ."
            historically_grown = []
        else:
            historically_grown = historical_crops['Crop'].unique()
            historically_grown_text = f'Crops historically grown in {season}: {", ".join(historically_grown)}'

        comparison_result = f"The predicted crop '{predicted_crop}' {'has' if predicted_crop in historically_grown else 'has NOT'} been historically grown in this period."

       
        response = {
            'predicted_crop': predicted_crop,
            'comparison_text': comparison_result,
            'historical_grown_text': historically_grown_text
        }

        print(f"Returning response: {response}") 
        return jsonify(response)

    except Exception as e:
        print(f"Error processing prediction: {e}")  
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
