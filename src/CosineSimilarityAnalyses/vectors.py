from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

class Vectors:

    def __init__(self, x):
        if isinstance(x, np.ndarray) and x.ndim == 2 and 
                        np.issubdtype(x.dtype, np.floating):            
            self.data = normalize(x)
            self.n_samples, self.n_features = np.shape(x)
        elif isinstance(x, Vectors):
            self.data = x.data
            self.n_samples = x.n_samples
            self.n_features = x.n_features
        else:
            raise ValueError("Invalid input type. "+
            "Expected a 2D numpy array of floating-point "+
            "numbers or an instance of Vectors.")

    def __getitem__(self, index):
        if len(index) == 2:
            return self.data[index[0], index[1]]
        elif len(index) == 1:
            return self.data[index[0]]
        else:
            raise ValueError("Invalid index. Expected a 1D or 2D index.")

    def __setitem__(self, index, value):
        if len(index) == 2:
            self.data[index[0], index[1]] = value
        elif len(index) == 1:
            self.data[index[0]] = value)
        