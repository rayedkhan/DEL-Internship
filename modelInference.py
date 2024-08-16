# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# load saved scaler, label encoder & autoencoder models
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
autoencoder = tf.keras.models.load_model('autoencoder_model.h5')

# load & pre-process new data (logs from csv file)
data = pd.read_csv('new_email_logs.csv')

features = data[['UUID', 'recipient', 'sender', 'subject', 'time_processed']]
uuids = features['UUID'] # UUIDs stored for retrieval later
features = features.drop(columns=['UUID'])

features['recipient'] = label_encoder.fit_transform(features['recipient'])
features['sender'] = label_encoder.fit_transform(features['sender'])
features['subject'] = label_encoder.fit_transform(features['subject'])
features['time_processed'] = pd.to_datetime(features['time_processed']).astype(np.int64) / 10**9

features_scaled = scaler.transform(features)

# perform inference using autoencoder model
reconstructed = autoencoder.predict(features_scaled)
reconstruction_error = np.mean(np.square(features_scaled - reconstructed), axis=1)
threshold = np.percentile(reconstruction_error, 95)  # threshold may be fine-tuned
anomalies = reconstruction_error > threshold

# map anomalies back to UUIDs
anomalous_uuids = data.loc[anomalies, 'UUID']

# print results
print(f'Number of anomalies detected: {np.sum(anomalies)}')
print('UUIDs of potentially harmful email logs:')
print(anomalous_uuids)

# histogram for distribution of reconstruction errors (relative to new data)
plt.figure(figsize=(10, 6))
sns.histplot(reconstruction_error, bins=50, kde=True)
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error with Anomaly Threshold on New Data')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# t-SNE visualization of anomalous email logs
features_with_errors = np.hstack(
    [features_scaled, reconstruction_error.reshape(-1, 1)])
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features_with_errors)

plot_data = pd.DataFrame(features_tsne, columns=['Dimension 1', 'Dimension 2'])
plot_data['Reconstruction Error'] = reconstruction_error
plot_data['Anomaly'] = anomalies

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=plot_data,
    x='Dimension 1',
    y='Dimension 2',
    hue='Anomaly',
    palette={
        0: 'blue',
        1: 'red'
    },  # blue for normal, red for anomalies
    legend='full',
    alpha=0.7,
    edgecolor=None)
plt.title('t-SNE Visualization of Email Logs')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Anomaly')
plt.show()

# projects high-dimensional email log features into two dimensions, allowing for easier visualization to identify patterns and clusters within data 
# highlights how well autoencoder model is distinguishing between normal data and anomalies by representing anomalies as red points and normal data as blue points 
# assess effectiveness of model in detecting potentially harmful emails based on separation and distribution in  lower-dimensional space
