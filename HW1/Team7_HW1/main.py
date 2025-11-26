import pandas as pd
import numpy as np

def customConv(data,c):
    x=data.copy()
    result = []
    for i in range(len(c)-1): # padding
      x.insert(0,0)
      x.append(0)
    for i in range(len(x)-len(c)+1): # dot product
      tempSum = 0
      for j in range(len(c)):
        tempSum += x[i+j]*c[len(c)-1-j] # flip and multiply
      result.append(tempSum)

    return result

def butterworth_lowpass_coefficients(fc, fs):
    nyq = 0.5 * fs
    norm_cutoff = fc/nyq
    wc = np.tan((3.14159) * norm_cutoff)  

    # second-order section coefficients (repeated to create a 4th-order filter)
    b0 = (wc**2)
    b1 = 2* (wc**2)
    b2 = (wc**2)
    a0 = (wc**2) + (2**0.5) *wc +1
    a1 = 2* ((wc**2) - 1)
    a2 = (wc**2) - (2**0.5) * wc + 1

    # cascade two second-order filters
    
    b_2nd_order = [b0/a0,b1/a0,b2/a0] #np.array([b0, b1, b2]) / a0
    a_2nd_order = [1, a1/a0, a2/a0]#np.array([1, a1 / a0, a2 / a0])

    # convolve the numerator and denominator 
    b = customConv(b_2nd_order, b_2nd_order)  # 4th-order numerator
    a = customConv(a_2nd_order, a_2nd_order)  # 4th-order denominator

    return b, a

# Function to apply the 4th-order filter using convolution
def apply_filter(data, b, a):
    # Apply the numerator (b) coefficients with convolution
    filtered_data = customConv(data, b)

    # subtract a from the output
    for i in range(4, len(data)):  
        filtered_data[i] -= (a[1] * filtered_data[i - 1] + 
                             a[2] * filtered_data[i - 2] +
                             a[3] * filtered_data[i - 3] +
                             a[4] * filtered_data[i - 4])

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
    - data: the accelerometer data
    - window_size: size of each sliding window
    - step_size: step size for the sliding window (overlap is window_size - step_size)
    - up_percentile: percentile for determining the upper threshold within a window 
    - low_percentile: percentile for determining the lower threshold within a window
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

    #print(f"Standard deviation of the data: {manual_std(data, mean)}")
    #print(f"high_variability_threshold for the data: {high_variability_threshold}")

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

    #print(f"Global upper threshold: {global_up_threshold}")
    #print(f"Global lower threshold: {global_low_threshold}")

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

    data = pd.read_csv(file_name, delimiter=",") # android data uses "," delimiter

    # Remove gravity from the x, y, and z axes
    filt_data_z = remove_gravity_full(data['z'])
    filt_data_x = remove_gravity_full(data['x'])
    filt_data_y = remove_gravity_full(data['y'])

    # Calculate the magnitude of the acceleration vector (with gravity removed)
    filtered_magnitude = [(x**2 + y**2 + z**2)**0.5 for x, y, z in zip(filt_data_x, filt_data_y, filt_data_z)]

    # Apply low-pass filter manually
    cutoff_frequency = 4 #Hz
    sampling_rate = interval #100Hz by default

    b, a = butterworth_lowpass_coefficients(cutoff_frequency, sampling_rate) # acquired numerator, denominator coefficients
    filtered_data = apply_filter(filtered_magnitude, b, a) # actual application of the filter( both backwards and forwards to avoid phase shift )

    # Peak detection
    peaks_full, local_up_ths, local_low_ths, global_up_th, global_low_th = peak_finding_3(filtered_data)

    step_count_full = len(peaks_full)

    print(f"Filtered Step Count is: {step_count_full}")


file_name = input("Please enter the file name: ")
count_steps(file_name)