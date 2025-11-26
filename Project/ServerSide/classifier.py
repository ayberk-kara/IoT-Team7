import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model
import constants

# Load the model
loaded_model = load_model('my_gru_model.h5')

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist= 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Filter coefficients
    y = filtfilt(b, a, data)  # Apply filter using filtfilt
    return y

def sensorLoggerCSVtotestDatasetCSV(sensorLoggerCSV):
    pd_accel = pd.read_csv(sensorLoggerCSV)
    magList = np.sqrt(pd_accel["x"]**2 + pd_accel["y"]**2 + pd_accel["z"]**2)
    cutoff_frequency = 5  # Desired cutoff frequency in Hz
    sampling_frequency = 100  # Sampling frequency in Hz (adjust this based on your data)
    filtered_magnitude = butter_lowpass_filter(magList, cutoff_frequency, sampling_frequency)
    magList = filtered_magnitude
    newDF = pd.DataFrame(columns=['magnitude'])
    for i in range(int(np.floor(len(magList)/350)) -1):
        new_row = pd.DataFrame([{'magnitude': magList[i*350:(i+2)*350]}])
        newDF = pd.concat([newDF, new_row], ignore_index=True)
    return newDF

def predict(file_name):
    
    df_server = sensorLoggerCSVtotestDatasetCSV(file_name)

    parsed_magnitude = []
    tempList = (df_server["magnitude"]).tolist()
    for i in range (len(df_server)):
            parsed_magnitude.append([float(x) for x in tempList[i]])
    df_server["magnitude"] = parsed_magnitude

    magnitude_data = df_server['magnitude']
    magnitude_data = np.array([np.array(mag) for mag in magnitude_data])  # Convert magnitude strings to arrays
    magnitude_data = magnitude_data.reshape((magnitude_data.shape[0], magnitude_data.shape[1], 1))  # Shape: (477, 800, 1)

    prediction = loaded_model.predict(magnitude_data)

    # Convert predictions to classes (for binary classification)
    predicted_classes = (prediction > 0.5).astype(int)

    for i in range(len(magnitude_data)):
        return predicted_classes[i][0]
    
def check_flatline(file_name):

    pd_accel = pd.read_csv(file_name)
    magList = np.sqrt(pd_accel["x"]**2 + pd_accel["y"]**2 + pd_accel["z"]**2)
    cutoff_frequency = 5  # Desired cutoff frequency in Hz
    sampling_frequency = 100  # Sampling frequency in Hz (adjust this based on your data)
    filtered_magnitude = butter_lowpass_filter(magList, cutoff_frequency, sampling_frequency)
    std_dev = np.std(filtered_magnitude)
    return std_dev < constants.FLATLINE_STD_DEV_THRESHOLD