import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from utils import *

class SeedInitiation:
    
    """
    class to generate the initial values (or seeds) of cluster centers
    """
    
    def __init__(self, n_clusters, seed_initiation, random_state):
        
        self.n_clusters = n_clusters
        self.seed_initiation = seed_initiation
        self.random_state = check_random_state(random_state)
        
    def generate_initial_cluster_centers(self, X, **cluster_init_params):        
        
        pass 
    
    def __recommended_method(self, X, anchor=None):
        
        # Creating a basis set spanning n_feature dimensional vector space        
        if anchor is None:
            basis = self.random_state.uniform(low=-1, high=1, 
                                            size=(self.n_features, 
                                                  self.n_features))
        else:
            basis = self.random_state.uniform(low=-1, high=1, 
                                            size=(self.n_features-1, 
                                                  self.n_features))
            basis = np.r_[np.reshape(anchor, (1,self.n_features)), basis]
            
        basis = normalize(basis)
        
        # Converting the basis into an orthonormal basis of same span
        orthonormal_basis = gram_schmidt(basis)
        # Transforming X in the space of new orthonormal basis
        transformed_X = np.dot(X, orthonormal_basis.T)
        
        proj_dirs = NCP_sq_lattice(self.n_feature, self.n_clusters)
        mat = np.dot(proj_dirs, transformed_X.T)
        seed_in_X_indices = np.c_[np.argmin(mat, axis=1), 
                                  np.argmax(mat, axis=1)].flatten()
        seed_in_X_indices = pd.unique(seed_in_X_indices)
        X0 = X[seed_in_X_indices]
        mat2 = 1-np.dot(X0, X0.T)[1:]
        prob = np.sum(mat2, axis=1)/np.sum(mat2)
        self.initial_cluster_centers = np.r_[[X0[0]], 
                                             X0[self.random_state.choice(
                                                 np.arange(len(prob))+1, 
                                                 size=self.n_clusters, 
                                                 replace=False, 
                                                 p=prob)]]
            

class Clustering(SeedInitiation):
    
    def __init__(self, n_clusters, max_iter=300, cost_tolerance=0.0001, 
                 seed_initiation='recommended', random_state=None):
        
        SeedInitiation.__init__(n_clusters, seed_initiation, random_state)
        self.max_iter = max_iter
        self.cost_tolerance = cost_tolerance
        
    def fit(self, X, **cluster_init_params):
        
        if not (isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating)):
            raise ValueError("X must be a floating numpy array")
        elif not np.allclose(np.linalg.norm(X, axis=1), 1):
            raise ValueError("X rows must be unit vectors")
        else:
            pass
            
        if np.shape(X) == 2:
            self.n_obs, self.n_features = np.shape(X)
        else:
            raise ValueError("X must be a 2D array")
        
        self.generate_initial_cluster_centers(X, **cluster_init_params)
        cluster_centers = self.initial_cluster_centers
        self.cost_per_iter = []
        
        for i in range(self.max_iter):
            d_sq = 1-np.dot(X, cluster_centers.T)
            cluster_assignments = np.argmin(d_sq, axis=1)
            cost = np.sum(np.arg(d_sq, axis=1))
            if cost <= self.cost_tolerance:
                self.cost_per_iter.extend([cost])
                break
            else:
                self.cost_per_iter.extend([cost])
            for c_id in range(self.n_clusters):
                cluster_centers[c_id] = np.sum(X[cluster_assignments==c_id], 
                                               axis=1)
            cluster_centers = normalize(cluster_centers)
            
        self.cluster_centers = cluster_centers
        self.cluster_assignments = cluster_assignments
        
    def predict(self, X):
        
        d_sq = 1-np.dot(X, self.cluster_centers.T)
        cluster_assignments = np.argmin(d_sq, axis=1)
        
        return cluster_assignments
    
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        