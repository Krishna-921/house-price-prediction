from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import gdown


app = Flask(__name__)
# Google Drive file ID for the model
MODEL_FILE_ID = "1vLXb8ARgPPyjfdK1LPSeBILBeNCV0AoJ"  # Replace with your actual file ID
MODEL_PATH = "best_house_price_model.pkl"

# Define the correct relative path
encoder_path = os.path.join("model", "label_encoder.pkl")

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)
        print("Download complete.")

# Ensure model is available before loading
download_model()

# Load the model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    
# Load the label encoder
try:
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    print("Label Encoder loaded successfully!")
except Exception as e:
    print(f"Error loading label encoder: {e}")

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
