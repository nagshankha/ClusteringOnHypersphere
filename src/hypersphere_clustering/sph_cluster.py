import numpy as np

class SeedInitiation:
    
    def __init__(self, n_clusters, seed_initiation, random_state):
        
        self.n_clusters = n_clusters

class Clustering(ClusterInitiation):
    
    def __init__(self, n_clusters, max_iter=300, cost_tolerance=0.0001, 
                 seed_initiation='recommended', random_state=None):
        
        SeedInitiation.__init__(n_clusters, seed_initiation, random_state)
        self.max_iter = max_iter
        self.cost_tolerance = cost_tolerance
        
    
        


        