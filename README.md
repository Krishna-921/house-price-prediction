# House Price Prediction

## 📌 Project Overview
This project is a **House Price Prediction Model** built using **Machine Learning** and deployed using **Render**. It predicts house prices based on various features such as median income, house age, and ocean proximity. The model is trained on the **California Housing Dataset** and is served using a **Flask API**.

## 🚀 Tech Stack
- **Programming Language:** Python
- **Machine Learning Model:** Scikit-learn
- **Web Framework:** Flask
- **Deployment Platform:** Render
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Pickle
- **Frontend:** HTML, CSS (Jinja2 Templates)

---
## 📊 Model Creation Process
### 1️⃣ Data Preprocessing
- Loaded the **California Housing Dataset**.
- Handled missing values and performed **feature engineering**.
- Encoded the categorical column **(`ocean_proximity`)** using Label Encoding.
- Split the dataset into **training** and **testing** sets.

### 2️⃣ Model Training
- Used **Linear Regression** for prediction.
- Evaluated the model using **RMSE (Root Mean Square Error)**.
- Saved the trained model as **`best_house_price_model.pkl`**.
- Saved the label encoder as **`label_encoder.pkl`**.

### 3️⃣ Flask API Development
- Created a **Flask application** to serve the model.
- Developed an **HTML frontend** for user input.
- Loaded the trained model and encoder dynamically from the `model/` directory.
- Used `gdown` to fetch large files from Google Drive if needed.

---
## 🔧 Deployment Process on Render
### 1️⃣ Push Code to GitHub
Ensure all necessary files are uploaded:
- `app.py` (Flask API)
- `train_model.py` (Model Training Script)
- `requirements.txt` (Dependencies)
- `templates/` (HTML Files)
- `model/` (Pickle Files, or load from Drive)
- `static/` (CSS, Images)

### 2️⃣ Create a Render Web Service
1. Go to [Render](https://render.com/).
2. Click **New Web Service** and connect your GitHub repository.
3. Select **Python 3.11** as the environment.
4. Set the **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
5. Set the **Start Command**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:10000 app:app
   ```
6. Click **Deploy**.

### 3️⃣ Handling Large Model Files
Since Render has file size limitations, use **Google Drive** for storing large `.pkl` files:
- Upload the model to Google Drive.
- Use `gdown` in `app.py` to fetch it dynamically:
  ```python
  import gdown
  url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
  output = "model/best_house_price_model.pkl"
  gdown.download(url, output, quiet=False)
  ```

---
## 🎯 Usage Instructions
1. **Run Locally**:
   ```bash
   python app.py
   ```
2. **Access the Web Interface** at `http://localhost:5000`.
3. Enter **house details** and get the **predicted price**.

---
## 🎉 Future Enhancements
- Improve model accuracy using advanced algorithms.
- Deploy using **Docker + AWS Lambda**.
- Add **database integration** for storing predictions.

🔗 **Live Demo:** [House Price Predictor on Render](https://your-app-url.render.com/)

