import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state

class SeedInitiation:
    
    """
    class to generate the initial values (or seeds) of cluster centers
    """
    
    def __init__(self, n_clusters, seed_initiation, random_state):
        
        self.n_clusters = n_clusters
        self.seed_initiation = seed_initiation
        self.random_state = check_random_state(random_state)
        
    def generate_cluster_centers(self, **kwargs):
        
        pass 
    
    def __recommended_method(self, anchor=None):
        
        # Create a basis set spanning n_feature dimensional vector space
        
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
        orthonormal_basis = gram_schmidt(basis)
        
        
        
            
            

class Clustering(SeedInitiation):
    
    def __init__(self, n_clusters, max_iter=300, cost_tolerance=0.0001, 
                 seed_initiation='recommended', random_state=None):
        
        SeedInitiation.__init__(n_clusters, seed_initiation, random_state)
        self.max_iter = max_iter
        self.cost_tolerance = cost_tolerance
        
    def fit(self, X):
        
        self.n_obs, self.n_features = np.shape(X)
        self.generate_cluster_centers()
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
    
        
def gram_schmidt(X):
    
    if np.shape(X) == 2:
        n_rows, n_columns = np.shape(X)
    else:
        raise ValueError("X must be a 2D array")
    
    if n_rows != n_columns:
        raise ValueError("X must be a square matrix")    
    elif np.isclose(np.linalg.det(X), 0):
        raise ValueError("X must not be a singular matrix")
    else:
        orthonormal_basis = np.ones(np.shape(X))
        
    for row in range(n_rows):
        if row == 0:
            orthonormal_basis[row] = X[row]/np.linalg.norm(X[row])
        else:
            orthonormal_basis[row] -= np.sum(np.dot(orthonormal_basis[:row], 
                                                    orthonormal_basis[row])*
                                             orthonormal_basis[:row].T, axis=1)
            orthonormal_basis[row] /= np.linalg.norm(orthonormal_basis[row])
            
    return orthonormal_basis
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        