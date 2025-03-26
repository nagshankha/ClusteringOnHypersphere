import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state

class SeedInitiation:
    
    def __init__(self, n_clusters, seed_initiation, random_state):
        
        self.n_clusters = n_clusters
        self.seed_initiation = seed_initiation
        
    def generate_cluster_centers(self, **kwargs):
        
        pass

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
    
        


        