import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from scipy.stats import truncnorm

class GeneratePointClusters:

    def __init__(self, cluster_centers, cluster_sizes, 
                            cluster_radii, random_state):
        self.cluster_centers = cluster_centers
        self.n_clusters, self.dim = np.shape(self.cluster_centers)
        self.cluster_sizes = cluster_sizes
        self.cluster_radii = cluster_radii
        self.random_state = check_random_state(random_state)

    def generate_points(self, truncate_at_radius = False):
        for clusterID in range(self.n_clusters):
            cluster_center = self.cluster_centers[clusterID]
            cluster_radius = self.cluster_radii[clusterID]
            cluster_size = self.cluster_sizes[clusterID]
            if truncate_at_radius:
                theta_r = truncnorm.rvs(0, cluster_radius, loc = 0, 
                                        scale = 0.5*cluster_radius, 
                                        size = cluster_size,
                                        seed = self.random_state)
            else:
                theta_r = self.random_state.normal(loc=0, scale=0.5*cluster_radius,
                                                   size=cluster_size)



    def __setattr__(self, name, value):

        pass

