import numpy as np
def scaled_norm(X):
    X_norm = (X-np.mean(X))/np.std(X)
    return X_norm