import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from scipy.stats import truncnorm

class GeneratePointClusters:

    def __init__(self, cluster_centers, cluster_sizes, 
                            cluster_radii, random_state):
        self.cluster_centers = cluster_centers
        self.n_clusters, self.dim = np.shape(self.cluster_centers) # dim cannot be less than 2.. add exception in __setattr__
        self.cluster_sizes = cluster_sizes
        self.cluster_radii = cluster_radii
        self.random_state = check_random_state(random_state)

    def generate_points(self, truncate_at_radius = True):
        points = []
        for clusterID in range(self.n_clusters):
            cluster_center = self.cluster_centers[clusterID]
            cluster_radius = self.cluster_radii[clusterID]
            cluster_size = self.cluster_sizes[clusterID]
            if truncate_at_radius:
                truncation_radius = cluster_radius
            else:
                truncation_radius = np.pi
            theta_r = truncnorm.rvs(0, truncation_radius, loc = 0, 
                                    scale = 0.5*cluster_radius, 
                                    size = cluster_size,
                                    seed = self.random_state)
            mat = self.random_state.uniform(
                    low = np.cos(theta_r)[:,None]/cluster_center[None,2:], 
                    high=1.0, size = (cluster_size, self.dim-2))
            c1 = 1 - np.sum(mat**2, axis=1)
            c2 = np.cos(theta_r) - np.sum(mat*cluster_center[2:], axis=1)
            sum12 = np.sum(cluster_center[:2]**2)
            x1 = c2/sum12
            x1 += (np.random_state.choice([-1,1], size=cluster_size) *
                    np.sqrt((c2**2*cluster_center[0]**2) - 
                            (sum12*(c2**2 - (c1*cluster_center[1]**2))))/sum12)
            x2 = (c2 - (x1*cluster_center[0]))/cluster_center[1]
            points.append(np.c_[clusterID, x1, x2, mat])

        points = np.vstack(points)

        if np.allclose(np.linalg.norm(points, axis=1), 1):
            return points
        else:
            raise RuntimeError("All generated points must lie on unit hypersphere")
                



    def __setattr__(self, name, value):

        pass

