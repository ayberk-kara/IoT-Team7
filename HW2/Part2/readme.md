Once prompted with the zipfile's location, "directory + filename" should be used:

/home/osboxes/Desktop/hw2_venv/UCI HAR Dataset.zip

After the dataset is read successfully, frames are formed, 9 columns of raw data and 2 columns for subjectID and label.

The formed frames are shuffled for better learning practices at each epoch.

In total, there are 4 different scenarios:

A) Training and Testing with default Train-Test Split, Collective Approach

B) Training and Testing with custom Train-Test Split, Collective Approach

C) Training and Testing with custom Train-Test Split, Personal Approach

D) Training and Testing with custom Train-Test Split and Transfer Learning for later layers for subject-based data, Personal Approach


These sequences run sequentially, obtained results are seperated by print statements and indicators.

For the case C, a specific CNN is used that is slightly more complex than the other scenarios. This happens to increase the learning of the model from the very few training data it has to go through.

A custom split was applied to the dataset to achieve better results. While the dataset contained appropriate values for general data, there were issues with the values for test and train data specifically related to personal data. After training, the model was further enhanced using transfer learning techniques to improve performance.
