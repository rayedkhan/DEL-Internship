# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib

# load & preprocess data (logs from CSV file)
data = pd.read_csv('email_logs.csv')

features = data[['UUID', 'recipient', 'sender', 'subject', 'time_processed']]
features = features.drop(columns=['UUID'])

label_encoder = LabelEncoder()
features['recipient'] = label_encoder.fit_transform(features['recipient'])
features['sender'] = label_encoder.fit_transform(features['sender'])
features['subject'] = label_encoder.fit_transform(features['subject'])

features['time_processed'] = pd.to_datetime(features['time_processed']).astype(np.int64) / 10**9

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# split data into training and testing sets
X_train, X_test = train_test_split(features_scaled, test_size=0.2, random_state=42)

# build autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 16

input_layer = Input(shape=(input_dim, ))

encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# train & save autoencoder model
history = autoencoder.fit(X_train,
                          X_train,
                          epochs=50,
                          batch_size=64,
                          validation_split=0.1,
                          verbose=1)
autoencoder.save('autoencoder_model.h5')

# plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# shows loss values for both training and validation sets
# useful to monitor how well the model is learning, and whether it's overfitting
# ideally, both loss values should decrease and converge, indicating good model performance
# edit autoencoder architecture (layers, # of neurons etc.) based on this for optimum training and fine-tuning

# perform inference on test data set & compute reconstruction error
reconstructed = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructed), axis=1)

# set threshold for anomaly classification
threshold = np.percentile(reconstruction_error, 95)  # can be fine-tuned

# histogram for distribution of reconstruction errors
plt.figure(figsize=(10, 6))
sns.histplot(reconstruction_error, bins=50, kde=True)
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error with Anomaly Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# histogram to visualize  distribution of reconstruction errors on test data set
# most data points should have low reconstruction errors, if they are similar to (normal) patterns the model has learned
# anomalies expected to have higher reconstruction errors, helps in their identification
# threshold line (in red) to visualize boundary separating normal and anomalous data points

# ensures consistent data scaling & mapping operations in inference model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
