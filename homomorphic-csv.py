# Kyle Mullen
# 4/11/25
# Processing provided CSV instead with PHE Library


import numpy as np
import pandas as pd
import phe as paillier

# Load data from local CSV file in the same directory
heart_data = pd.read_csv('heart.csv')

# Extract only the features we need for the model
selected_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X = heart_data[selected_features]

# Remove rows with missing values if any
X = X.dropna()

# Define the prediction model coefficients
coefficients = {
    'age': 0.5581,
    'trestbps': 0.0048,
    'chol': 0.0044,
    'thalach': -0.0036,
    'oldpeak': 0.1290,
    'intercept': -28.9796
}

# Function to make prediction using the provided model
def predict(row):
    # f(·) = 0.5581 × age + 0.0048 × trestbps + 0.0044 × chol − 0.0036 × thalach + 0.1290 × oldpeak − 28.9796
    result = (coefficients['age'] * row['age'] + 
              coefficients['trestbps'] * row['trestbps'] + 
              coefficients['chol'] * row['chol'] + 
              coefficients['thalach'] * row['thalach'] + 
              coefficients['oldpeak'] * row['oldpeak'] + 
              coefficients['intercept'])
    return result

# Generate public and private keys for Paillier encryption
public_key, private_key = paillier.generate_paillier_keypair()

# Function to encrypt the features
def encrypt_features(data_row):
    encrypted_dict = {}
    for key, value in data_row.items():
        encrypted_dict[key] = public_key.encrypt(float(value))
    return encrypted_dict

# Function to compute the encrypted prediction
def encrypted_predict(encrypted_features):
    # Initialize encrypted prediction with the intercept
    encrypted_result = public_key.encrypt(coefficients['intercept'])
    
    # Add each term to the prediction
    for feature, coefficient in coefficients.items():
        if feature != 'intercept':
            # Multiply encrypted feature with coefficient
            weighted_feature = encrypted_features[feature] * coefficient
            # Add to the encrypted result
            encrypted_result += weighted_feature
    
    return encrypted_result

# Example usage
print("Sample of original data:")
sample_data = X.head(5)
print(sample_data)

print("\nPredictions on plaintext data:")
for i, row in sample_data.iterrows():
    prediction = predict(row)
    print(f"Patient {i}: {prediction:.4f}")

print("\nPredictions using homomorphic encryption:")
for i, row in sample_data.iterrows():
    # Encrypt the features
    encrypted_row = encrypt_features(row)
    
    # Make prediction on encrypted data
    encrypted_prediction = encrypted_predict(encrypted_row)
    
    # Decrypt the result
    decrypted_prediction = private_key.decrypt(encrypted_prediction)
    
    print(f"Patient {i}: {decrypted_prediction:.4f}")