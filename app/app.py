from flask import Flask, request, jsonify, render_template
import pandas as pd
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load saved models dynamically
models = {}
model_dir = "../models"
for filename in os.listdir(model_dir):
    if filename.endswith(".h5"):
        station_name = filename.split("_")[-1].replace(".h5", "")
        models[station_name] = tf.keras.models.load_model(
        os.path.join(model_dir, filename), 
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
        )
@app.route('/')
def home():
    return render_template('index.html')  # Frontend dashboard

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['csv_file']
    df = pd.read_csv(file)

    predictions = {}
    for station, model in models.items():
        if station in df.columns:
            X_input = df[[f'SPI3_t-{i}' for i in range(1, 4)]].values
            pred_values = model.predict(X_input)
            predictions[station] = pred_values.flatten().tolist()

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
