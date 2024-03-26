from flask import Flask, render_template, request
import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import send_file
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
from flask import jsonify
import json
import base64

app = Flask(__name__)

uploads_dir = "./instance/upload"
dataset = ""

# Load model
bandwidth_model = load_model("models/bandwidth_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(uploads_dir, f.filename))
        dataset = pd.read_csv(os.path.join(uploads_dir, f.filename))
        # print(data_preprocessing(dataset))
        return 'file uploaded successfully'

@app.route('/forecast')
def predict_bandwidth():
    global dataset, forecast_values
    dataset, max_traffic, min_traffic = data_preprocessing(dataset)
    # Extract traffic data from dataset
    traffic_data = dataset['Traffic Total (Volume)'].values.astype('float32')

    # Normalize traffic data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_traffic_data = scaler.fit_transform(traffic_data.reshape(-1, 1))

    # Organize data into input sequences
    history_length = 1
    input_sequence = create_input_sequence(normalized_traffic_data, history_length)

    # Get the value of future_steps from the query parameters
    future_steps = int(request.args.get('future_steps', 3))  # Default to 3 if not provided
    forecast = predict_future(bandwidth_model, input_sequence, future_steps)

    # Inverse transform forecasted data
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Extract the last date from the dataset
    last_date = dataset['Date Time'].iloc[-1]
    last_date = pd.to_datetime(last_date).strftime('%Y-%m-%d %H:%M:%S')
    last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')

    # Generate predicted dates for the next future_steps
    start_date = last_date.replace(hour=0, minute=0, second=0) + datetime.timedelta(days=1)
    predicted_dates = [start_date + datetime.timedelta(days=i) for i in range(future_steps)]

    forecast_values = [float(value[0]) for value in forecast]

    # Read the image file as bytes
    with open('./instance/images/bandwidth_forecast.png', 'rb') as f:
        image_blob = f.read()

    # Encode the image blob as base64
    image_base64 = base64.b64encode(image_blob).decode('utf-8')

    # Return forecast values along with the encoded image and max/min traffic values
    return jsonify({'forecast': forecast_values, 'imageBlob': image_base64, 'maxTraffic': max_traffic, 'minTraffic': min_traffic})




@app.route('/plot')
def plot_graph():
    global dataset
    if dataset is not None:
        dataset = data_preprocessing(dataset)
        
        # Plotting
        plt.figure(figsize=(7, 7))
        plt.plot(dataset['Date Time'], dataset['Traffic Total (Volume)'], marker='o', linestyle='-')
        plt.xticks(rotation=45)
        plt.xlabel('Date Time')
        plt.ylabel('Traffic Volume')
        plt.title('Traffic Volume Over Time')
        plt.grid(True)
        plt.savefig('./instance/images/traffic_volume.png')
        
        # Return path to the image file
        return send_file('./instance/images/traffic_volume.png', mimetype='image/png')
    else:
        return 'No dataset available'
    
def data_preprocessing(dataset):
    X = dataset.iloc[:, 0:10]
    Y = dataset.iloc[:, 0:10]
    dataset = dataset.drop(dataset.tail(2).index)
    dataset['Date Time'] = pd.to_datetime(dataset['Date Time'].str.split(' - ', expand=True)[0], format='%m/%d/%Y %I:%M:%S %p')
    dataset['Time'] = dataset['Date Time'].dt.hour + dataset['Date Time'].dt.minute / 60 + dataset['Date Time'].dt.second / 3600

    dataset = dataset.drop(['Traffic In (Speed)(RAW)','Traffic Total (Speed)','Traffic Out (Volume)', 'Traffic In (Volume)', 'Time','Date Time(RAW)','Traffic Total (Volume)(RAW)','Traffic Total (Volume)(RAW)','Traffic Total (Speed)(RAW)','Traffic In (Volume)(RAW)','Traffic In (Speed)','Traffic Out (Volume)(RAW)','Traffic Out (Speed)','Traffic Out (Speed)(RAW)','Downtime','Downtime(RAW)','Coverage','Coverage(RAW)'], axis=1)

    # Remove "MB" suffix from volume values and commas from the numeric values
    dataset['Traffic Total (Volume)'] = dataset['Traffic Total (Volume)'].replace({' MB': '', ',': ''}, regex=True).astype(float)

    # Calculate the mean of non-missing values
    mean_volume = dataset['Traffic Total (Volume)'].mean()

    # Replace missing values with the mean
    dataset.loc[dataset['Traffic Total (Volume)'].isnull(), 'Traffic Total (Volume)'] = mean_volume

    # Round values to the nearest integer
    dataset['Traffic Total (Volume)'] = dataset['Traffic Total (Volume)'].round(0)

    # Calculate the total traffic volume
    total_traffic_volume = dataset['Traffic Total (Volume)'].sum()
    print("Total Traffic Volume:", total_traffic_volume)
    
    
    max_traffic = dataset['Traffic Total (Volume)'].max()
    min_traffic = dataset['Traffic Total (Volume)'].min()
    print("Max Traffic Volume:", max_traffic)
    print("Min Traffic Volume:", min_traffic)

    return dataset, max_traffic, min_traffic
    
# Function to create input sequence
def create_input_sequence(data, history_length=1):
    input_sequence = data[-history_length:].reshape(1, history_length, 1)
    return input_sequence

# Function to predict future steps
def predict_future(model, input_sequence, future_steps):
    forecast = []
    current_sequence = input_sequence.copy()
    for _ in range(future_steps):
        future_data = model.predict(current_sequence)[0, 0]
        forecast.append(future_data)
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = future_data
    return forecast


if __name__ == '__main__':
    app.run(debug=True)
