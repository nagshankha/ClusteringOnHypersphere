from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

class Vectors:

    def __init__(self, x):
        if isinstance(x, np.ndarray) and (x.ndim == 2) and 
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
        
    def __setattr__(self, name, value):

        if name == 'data':
            if isinstance(x, np.ndarray) and (x.ndim == 2) and 
                        np.issubdtype(x.dtype, np.floating) and
                        np.allclose(np.linalg.norm(x, axis=1), 1):
                self.__dict__[name] = x
            else:
                raise ValueError("data attribute of Vectors instance "+
                "must be a 2D numpy array with each row being an unit vector.")

        elif name == 'n_samples':
            if value == np.shape(self.data)[0]:
                self.__dict__[name] = value
            else:
                raise ValueError("n_samples attribute of Vectors instance "+
                "must be equal to the number of rows in the data attribute.")

        elif name == 'n_features':
            if value == np.shape(self.data)[1]:
                self.__dict__[name] = value
            else:
                raise ValueError("n_features attribute of Vectors instance "+
                "must be equal to the number of columns in the data attribute.")

        else:
            self.__dict__[name] = value