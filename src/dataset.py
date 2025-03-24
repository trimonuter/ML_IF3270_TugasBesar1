# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

n_samples = 10000

# Load data from OpenML
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Shuffle the dataset
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])

# Apply the permutation and take only the first `n_samples`
X = X[permutation][:n_samples]
y = y[permutation][:n_samples]

# Reshape to maintain compatibility
X = X.reshape((X.shape[0], -1))