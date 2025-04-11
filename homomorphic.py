# Kyle Mullen
# 4/11/25
# Processing UCIMLREPO with PHE Library

from phe import paillier
import ucimlrepo
import numpy as np
import pandas as pd


# Fetch heart disease dataset from UCI ML Repository
heart_disease = ucimlrepo.fetch_ucirepo(id=45)
  
# Get features and targets
features = heart_disease.data.features
targets = heart_disease.data.targets

# Extract only the features we need for the model
selected_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X = features[selected_features]

# Remove rows with missing values
X = X.dropna()