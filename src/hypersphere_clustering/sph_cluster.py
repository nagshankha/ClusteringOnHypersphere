import numpy as np
import pandas as pd
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
        
    def generate_initial_cluster_centers(self, X, cluster_init_params):
        
        
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
        seed_in_X_indices = pd.unique(seed_in_X_indices)[:self.n_clusters]
        self.initial_cluster_centers = X[seed_in_X_indices]
            

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
        
        self.generate_initial_cluster_centers(X, cluster_init_params)
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
    
    """
    This function convert any arbitrary basis into an orthonormal basis of the 
    same span using Gram-Schmidt process
    """
    
    if not (isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating)):
        raise ValueError("X must be a floating numpy array")
    else:
        pass
    
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

def NCP_sq_lattice(n, m):
    
    """
    This function list m non-collinear points on n-dimensional square lattice
    """
    
    from sympy.utilities.iterables import multiset_permutations, permute_signs
    
    num = 0
    
    arr0 = np.zeros((n,n))
    arr0[np.tril_indices(n)] = 1
    
    arr1 = []
    for i in range(n-1):
        arr1.append(arr0[i]+arr0[(i+1):])
    arr1 = np.concatenate(arr1)
    
    def func(arr0, arr1, f):
        a = []
        for r in arr0:
            a.append((f*r)+arr1)
        a = np.concatenate(a)
        a = np.unique(a, axis=0)
        return a
    
    step = 0; arr = []
    while(num<=m):
        if step == 0:
            step_arr = arr0
        elif step == 1:
            step_arr = arr1
        else:
            step_arr = func(arr0, arr1, step-1)
        for r in step_arr:
            mp = list(multiset_permutations(list(r)))
            for mp_i in mp:
                p = list(permute_signs(mp_i))
                arr += p[:int(len(p)/2)]
                
        step = step+1
        num = len(arr)
        
    return np.array(arr[:m])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        