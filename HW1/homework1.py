

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Manually implemented Low-Pass Filter
def LowPassFilter(data, window_size):
    filtered_data = []
    for i in range(len(data)):
        if i < window_size:
            filtered_data.append(sum(data[:i+1]) / len(data[:i+1]))
        else:
            window = data[i - window_size:i]
            filtered_data.append(sum(window) / window_size)
    return filtered_data

# Manually implemented mean function
def manual_mean(window):
    total = sum(window)
    return total / len(window)

# Manually implemented standard deviation function
def manual_std(window, mean):
    variance = sum((x - mean)**2 for x in window) / len(window)
    return variance**0.5

# Manually remove gravity by subtracting the mean of the data
def remove_gravity_full(data):
    mean_gravity = manual_mean(data)
    return [value - mean_gravity for value in data]

# Manually implemented peak detection using sliding window
def peak_finding_3(data, window_size=100, step_size=50,
                   up_percentile=85, low_percentile=40,
                   global_up_threshold=None, global_low_threshold=None,
                   minimum_distance=25):

    """
    Finds peaks in the data using a sliding window approach with dynamically
    determined thresholds and global thresholds as a fallback mechanism for noise suppression.

    Parameters:
    - data: the accelerometer data (1D numpy array or list)
    - window_size: size of each sliding window
    - step_size: step size for the sliding window (overlap is window_size - step_size)
    - up_percentile: percentile for determining the upper threshold within a window (e.g., 85 for 85th percentile)
    - low_percentile: percentile for determining the lower threshold within a window (e.g., 40 for 40th percentile)
    - global_up_threshold: an absolute global upper threshold to ignore small peaks in periods of inactivity
    - global_low_threshold: an absolute global lower threshold for resetting during inactivity periods
    - minimum_distance: minimum distance between consecutive peaks (in number of samples)

    Returns:
    - List of indices where peaks are detected
    - List of upper and lower threshold values per window for plotting
    """

    # To make threshold calculation more responsive to changes in variability use segmentation
    segment_std_devs = []
    for i in range(0, len(data), step_size):
        segment = data[i:i+window_size]
        if len(segment) == window_size:
            mean_segment = manual_mean(segment)
            std_segment = manual_std(segment, mean_segment)
            segment_std_devs.append(std_segment)

    mean_variability = manual_mean(segment_std_devs)
    std_variability = manual_std(segment_std_devs, mean_variability)

    # Calculate the high variability threshold
    high_variability_threshold = mean_variability + 2.5 * std_variability


    hard_limit = 1
    steps = []  # to store the indices of detected peaks
    temp_peak = None  # Temporarily store a peak until validated
    reset = True  # Indicates whether we are ready to register a new peak
    local_upper_thresholds = []  # Store local upper thresholds for plotting
    local_lower_thresholds = []  # Store local lower thresholds for plotting
    window_positions = []  # Store positions of each window for plotting
    mean = manual_mean(data)

    print(f"Standard deviation of the data: {manual_std(data, mean)}")
    print(f"high_variability_threshold for the data: {high_variability_threshold}")

    if manual_std(data, mean) > high_variability_threshold:

      global_up_threshold = sorted(data)[int(0.70 * len(data))]   # Lower threshold for variable data
      global_low_threshold = sorted(data)[int(0.45 * len(data))]
    else:
      global_up_threshold = sorted(data)[int(0.80 * len(data))]   # Higher threshold for stable data
      global_low_threshold = sorted(data)[int(0.50 * len(data))]




    # Calculate global thresholds if not provided
    if global_up_threshold is None:
        global_up_threshold = sorted(data)[int(0.80 * len(data))]  # 80th percentile
    if global_low_threshold is None:
        global_low_threshold = sorted(data)[int(0.50 * len(data))]  # 50th percentile

    print(f"Global upper threshold: {global_up_threshold}")
    print(f"Global lower threshold: {global_low_threshold}")

    # Loop over the data in steps, using a sliding window
    for start in range(0, len(data) - window_size + 1, step_size):
        window_data = data[start:start + window_size]

        # Calculate dynamic window-based thresholds
        local_up_threshold = sorted(window_data)[int(0.85 * len(window_data))]
        local_low_threshold = sorted(window_data)[int(0.40 * len(window_data))]

        # Store local thresholds and window position for later plotting
        local_upper_thresholds.append((start, start + window_size, local_up_threshold))
        local_lower_thresholds.append((start, start + window_size, local_low_threshold))
        window_positions.append(start)

        for i in range(1, len(window_data) - 1):
            global_index = start + i

            # Check if it's a local peak
            if window_data[i] > window_data[i - 1] and window_data[i] > window_data[i + 1]:
                if window_data[i] > local_up_threshold and window_data[i] > global_up_threshold and reset and (window_data[i] > hard_limit):
                    if len(steps) == 0 or (global_index - steps[-1] >= minimum_distance):
                        if temp_peak is None:
                            temp_peak = global_index
                        elif data[global_index] > data[temp_peak]:
                            temp_peak = global_index

            if window_data[i] < global_low_threshold and window_data[i] < local_low_threshold:
                reset = True
                if temp_peak is not None:
                    steps.append(temp_peak)
                    temp_peak = None

    if temp_peak is not None:
        steps.append(temp_peak)
    return steps, local_upper_thresholds, local_lower_thresholds, global_up_threshold, global_low_threshold

def count_steps(file_name, interval=100.0):

    data = pd.read_csv(file_name, delimiter=",")

    # Remove gravity from the x, y, and z axes
    filt_data_z = remove_gravity_full(data['z'])
    filt_data_x = remove_gravity_full(data['x'])
    filt_data_y = remove_gravity_full(data['y'])

    # Calculate the magnitude of the acceleration vector (with gravity removed)
    filtered_magnitude = [(x**2 + y**2 + z**2)**0.5 for x, y, z in zip(filt_data_x, filt_data_y, filt_data_z)]

    # Apply low-pass filter manually
    filtered_data = LowPassFilter(filtered_magnitude, window_size=13)

    # FFT using the library (NumPy)
    sampling_rate = interval  # Example, adjust to your actual sensor sampling rate
    fft_result_original = np.fft.fft(filtered_magnitude)
    n = len(filtered_magnitude)
    frequencies = np.fft.fftfreq(n, 1 / sampling_rate)
    magnitude_original = np.abs(fft_result_original)

    # FFT of filtered data
    fft_result_filtered = np.fft.fft(filtered_data)
    magnitude_filtered = np.abs(fft_result_filtered)

    # Plot FFT results
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(frequencies[:len(frequencies)//2], magnitude_original[:len(magnitude_original)//2])
    plt.title("Original Frequency Spectrum (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(frequencies[:len(frequencies)//2], magnitude_filtered[:len(magnitude_filtered)//2])
    plt.title("Filtered Frequency Spectrum (FFT, 5 Hz LPF)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Peak detection after filtering
    peaks_full, local_up_ths, local_low_ths, global_up_th, global_low_th = peak_finding_3(filtered_data)

    step_count_full = len(peaks_full)

    print(f"Filtered Step Count is: {step_count_full}")

    plt.figure(figsize=(16, 6))

    plt.axhline(global_up_th, color='red', linestyle='--', label='Global Upper Threshold')
    plt.axhline(global_low_th, color='blue', linestyle='--', label='Global Lower Threshold')

    for (start, end, th) in local_up_ths:
        plt.plot([start, end], [th, th], color='green', linestyle='--', label='Local Upper Threshold' if start == 0 else "")

    for (start, end, th) in local_low_ths:
        plt.plot([start, end], [th, th], color='purple', linestyle='--', label='Local Lower Threshold' if start == 0 else "")

    plt.plot(filtered_data, label="Accelerometer_data")
    plt.plot(peaks_full, [filtered_data[i] for i in peaks_full], "x", label='Step_full')
    plt.title("Step Counting_full with Thresholds")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    plt.show()


file_name = input("Please enter the file name: ")
count_steps(file_name)
