import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the historical crop data
historical_data = pd.read_csv('historical_crops_data.csv')  # Ensure this CSV is structured as Year, Month, Crop, Season

# Load the trained model
model = pickle.load(open('crop_prediction_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form or the simulator
    data = request.get_json()
    print(f"Received data: {data}")  # Debugging step to check received data in the server log

    try:
        # Extracting input values
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        season = data['season']
          # Assuming current year for historical comparison

        # Prepare the features for prediction
        final_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(final_features)
        predicted_crop = prediction[0]
        print(f"Predicted crop: {predicted_crop}")  # Debugging predicted crop

        # Get historical crops for comparison
        historical_crops = historical_data[(historical_data['Season'] == season) ]

        if historical_crops.empty:
            historically_grown_text = f"No crops historically grown in {season} ."
            historically_grown = []
        else:
            historically_grown = historical_crops['Crop'].unique()
            historically_grown_text = f'Crops historically grown in {season}: {", ".join(historically_grown)}'

        comparison_result = f"The predicted crop '{predicted_crop}' {'has' if predicted_crop in historically_grown else 'has NOT'} been historically grown in this period."

        # Return the prediction results
        response = {
            'predicted_crop': predicted_crop,
            'comparison_text': comparison_result,
            'historical_grown_text': historically_grown_text
        }

        print(f"Returning response: {response}")  # Debugging response data
        return jsonify(response)

    except Exception as e:
        print(f"Error processing prediction: {e}")  # Log any errors that occur
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
