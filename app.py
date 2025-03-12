from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join('model', '/Users/apple/PycharmProjects/House-Price-Prediction/model/best_house_price_model.pkl')
encoder_path = os.path.join('model', '/Users/apple/PycharmProjects/House-Price-Prediction/model/label_encoder.pkl')

# Load the trained model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['longitude']),
            float(request.form['latitude']),
            float(request.form['housing_median_age']),
            float(request.form['total_rooms']),
            float(request.form['total_bedrooms']),
            float(request.form['population']),
            float(request.form['households']),
            float(request.form['median_income']),
        ]

        # Encode 'ocean_proximity' using the loaded LabelEncoder
        ocean_proximity = request.form['ocean_proximity']
        encoded_ocean_proximity = label_encoder.transform([ocean_proximity])[0]
        features.append(encoded_ocean_proximity)

        # Prediction
        prediction = model.predict([features])[0]
        formatted_prediction = f"${prediction:,.2f}"

        return render_template('index.html', predicted_price=formatted_prediction)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
