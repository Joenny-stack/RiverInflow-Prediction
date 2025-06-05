from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import scipy.stats as st

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

def calculate_spi_gamma(series, scale=3):
    rolled = series.rolling(window=scale).sum().dropna()
    q = len(rolled[rolled == 0]) / len(rolled)
    rolled_positive = rolled[rolled > 0]
    if len(rolled_positive) < 2:
        return pd.Series(np.nan, index=rolled.index)
    shape, loc, scale_param = st.gamma.fit(rolled_positive, floc=0)
    cdf_values = np.zeros(len(rolled))
    for i, value in enumerate(rolled.values):
        if value == 0:
            cdf_values[i] = q
        else:
            cdf_values[i] = q + (1 - q) * st.gamma.cdf(value, shape, loc=loc, scale=scale_param)
    spi_values = st.norm.ppf(cdf_values)
    return pd.Series(spi_values, index=rolled.index)

@app.route('/')
def home():
    return render_template('index.html')  # Frontend dashboard

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Render upload/predict page
        return render_template('predict.html')
    
    file = request.files['csv_file']
    df = pd.read_csv(file)

    # Assume columns: Date, Rainfall, Station (optional)
    df.columns = df.columns.str.strip()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date')
    # Accept common rainfall column names
    rainfall_col = None
    for col in df.columns:
        if col.lower() in ['rainfall', 'precipitation', 'Precipitation (mm)', 'precipitation (mm)', 'rain', 'rain_mm']:
            rainfall_col = col
            break
    if rainfall_col is not None and rainfall_col != 'Rainfall':
        df = df.rename(columns={rainfall_col: 'Rainfall'})
    if 'Rainfall' not in df.columns:
        return jsonify({'error': 'CSV must contain a Rainfall or Precipitation column.'}), 400
    # Calculate SPI3
    df['SPI3'] = calculate_spi_gamma(df['Rainfall'], scale=3)
    # Add lag features
    for i in range(1, 4):
        df[f'SPI3_t-{i}'] = df['SPI3'].shift(i)
    df = df.dropna()
    if df.empty:
        return jsonify({'error': 'Not enough data to calculate SPI and lag features.'}), 400
    # Prepare input for model(s)
    predictions = {}
    for station, model in models.items():
        # If station column exists, filter; else, use all
        if 'Station' in df.columns and station not in df['Station'].unique():
            continue
        station_df = df if 'Station' not in df.columns else df[df['Station'] == station]
        if len(station_df) == 0:
            continue
        # Only predict if all lag columns exist and are not NaN
        if not all(col in station_df.columns for col in [f'SPI3_t-1', f'SPI3_t-2', f'SPI3_t-3']):
            continue
        X_input = station_df[[f'SPI3_t-1', f'SPI3_t-2', f'SPI3_t-3']].dropna()
        if X_input.empty:
            continue
        pred_values = model.predict(X_input.values)
        predictions[station] = pred_values.flatten().tolist()
    if not predictions:
        # Return error in a way the frontend can display
        return jsonify({'error': 'No predictions could be made for the uploaded data. Ensure your file has enough rows and valid rainfall/precipitation data.'}), 200
    return jsonify(predictions)

@app.route('/<filename>')
def serve_prediction_csv(filename):
    if filename.endswith('.csv'):
        return send_from_directory('..', filename)
    return "Not found", 404

if __name__ == '__main__':
    app.run(debug=True)
