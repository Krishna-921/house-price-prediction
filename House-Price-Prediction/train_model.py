import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle

# Create 'model' folder if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load the dataset
data = pd.read_csv('housing.csv')

# Handle missing values
data.fillna(data.select_dtypes(include='number').median(), inplace=True)

# Encode 'ocean_proximity' using LabelEncoder
le = LabelEncoder()
data['ocean_proximity'] = le.fit_transform(data['ocean_proximity'])

# Define features and target
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

# Save the trained model
with open('model/best_house_price_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the label encoder
with open('model/label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)
