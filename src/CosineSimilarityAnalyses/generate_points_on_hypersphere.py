import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from scipy.stats import truncnorm
from vectors import Vectors

class GeneratePointClusters:

    def __init__(self, cluster_centers, cluster_sizes, 
                            cluster_radii, random_state):
        self.cluster_centers = cluster_centers
        self.cluster_sizes = cluster_sizes
        self.cluster_radii = cluster_radii
        self.random_state = check_random_state(random_state)

    def generate_points(self, truncate_at_radius = True) -> Vectors:
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
            if clusterID == 0:
                points = Vectors(np.c_[x1, x2, mat], 
                                 clusterID = clusterID*np.ones(cluster_size, 
                                                               dtype=int),
                                 pointID = np.arange(cluster_size))
            else:
                points += Vectors(np.c_[x1, x2, mat], 
                                  clusterID = clusterID*np.ones(cluster_size, 
                                                               dtype=int),
                                  pointID = np.arange(cluster_size))

        return points    

    def __setattr__(self, name, value):

        if name == 'cluster_centers':
            if isinstance(value, np.ndarray) and (value.ndim == 2) and 
                        value.shape[1] >= 2 and 
                        np.issubdtype(value.dtype, np.floating) and
                        np.allclose(np.linalg.norm(value, axis=1), 1):
                self.__dict__[name] = value
                self.n_clusters, self.dim = np.shape(value)
            else:
                raise ValueError("cluster_centers attribute of GeneratePointClusters "+
                "must be a 2D numpy array with each row being an unit vector. "+
                "Each vector must also have more than or equal to 2 features "+
                "(or dimensions)")

        elif name == 'n_clusters':
            if value == np.shape(self.cluster_centers)[0]:
                self.__dict__[name] = value
            else:
                raise ValueError("n_clusters attribute of GeneratePointClusters "+
                "must be equal to the number of rows in the cluster_centers attribute.")

        elif name == 'dim':
            if value == np.shape(self.cluster_centers)[1]:
                self.__dict__[name] = value
            else:
                raise ValueError("dim attribute of GeneratePointClusters "+
                "must be equal to the number of columns in the cluster_centers attribute.")

        elif name == 'cluster_sizes':
            if isinstance(value, np.ndarray) and (value.ndim == 1) and 
                        np.issubdtype(value.dtype, np.integer):
                self.__dict__[name] = value
            else:
                raise ValueError("cluster_sizes attribute of GeneratePointClusters "+
                "must be a 1D numpy array of integers.")

        elif name == 'cluster_radii':
            if isinstance(value, np.ndarray) and (value.ndim == 1) and 
                        np.issubdtype(value.dtype, np.floating):
                if np.all(value<np.pi):
                    self.__dict__[name] = value
                else:
                    raise ValueError("All cluster radii must be less than pi.")
            else:
                raise ValueError("cluster_radii attribute of GeneratePointClusters "+
                "must be a 1D numpy array of floats.")

        elif name == 'random_state':
            if isinstance(value, np.random.RandomState):
                self.__dict__[name] = value
            else:
                raise ValueError("random_state attribute of GeneratePointClusters "+
                "must be an instance of np.random.RandomState.")

        else:
            self.__dict__[name] = value


