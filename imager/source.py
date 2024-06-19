import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pykronecker as kn

class DistributedSource():

    def __init__(self, src_str, src_dist=None):
        self.src_str = src_str
        self.src_dist=src_dist
    def __add__(self, src):
        assert isinstance(src, DistributedSource)
        temp_src = DistributedSource(src.src_str+self.src_str)
        assert src.src_dist is not None
        temp_src.src_dist = src.src_dist+self.src_dist
        return temp_src
    
    #This magic method is necessary for sum(list) to work
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @property
    def src_str(self):
        return self._src_str
    
    @src_str.setter
    def src_str(self, strength):
        self._src_str = strength

    @property
    def src_loc(self):
        return self._src_loc

    @src_loc.setter
    def src_loc(self, loc):
        self._src_loc = loc

    @property
    def src_dist(self):
        return self._src_dist
    
    @src_dist.setter
    def src_dist(self, dist):
        self._src_dist = dist

    def get_src_dist(self, vox_ctr):
        if self.src_dist is not None:
            return self.src_dist
        else:
            raise NotImplementedError



class GaussianSource(DistributedSource):
    
    def __init__(self, src_str, src_loc, src_cov):
        super().__init__(src_str)
        self.src_cov = src_cov
        self.src_loc = src_loc

    @property
    def src_cov(self):
        return self._src_cov

    @src_cov.setter
    def src_cov(self, cov):
        self._src_cov = cov

    def get_src_dist(self, vox_ctr):
        mult_gauss = multivariate_normal(
            mean = self.src_loc,
            cov=self.src_cov)
        self.src_dist = mult_gauss.pdf(x=vox_ctr)
        #The following line takes care of 2D voxel space case
        if self.src_dist.shape != vox_ctr.shape[:-1]:
            self.src_dist = self.src_dist.reshape(vox_ctr.shape[:-1])
        diff = vox_ctr-self.src_loc
        inv_cov = np.linalg.inv(self.src_cov)
        mahalanobis_dist = \
            np.sqrt(
                np.sum(
                    np.einsum(
                        "ijkl, lm -> ijkm", diff, inv_cov) * diff, axis=-1)
                    )
        #Zero out too weak voxel values
        self.src_dist[mahalanobis_dist>3] = 0
        self.src_dist /= self.src_dist.sum()
        #all the src strength are stored in Ci units
        #when calculating the forward projecion we convert it to Bq
        self.src_dist *= self.src_str 

        return self.src_dist

class CuboidGaussianSource(DistributedSource):
    
    def __init__(self, src_str, src_loc, src_cov, src_widths):
        super().__init__(src_str)
        self.src_cov = src_cov
        self.src_widths = src_widths
        self.src_loc = src_loc

    @property
    def src_cov(self):
        return self._src_cov

    @src_cov.setter
    def src_cov(self, cov):
        self._src_cov = cov
    @property
    def src_widths(self):
        return self._src_widths

    @src_widths.setter
    def src_widths(self, src_widths):
        self._src_widths = src_widths
        
    def get_src_dist(self, vox_ctr):
        mult_gauss = multivariate_normal(
            mean = self.src_loc,
            cov=self.src_cov)
        self.src_dist = mult_gauss.pdf(x=vox_ctr)
        #The following line takes care of 2D voxel space case
        if self.src_dist.shape != vox_ctr.shape[:-1]:
            self.src_dist = self.src_dist.reshape(vox_ctr.shape[:-1])
        diff = vox_ctr-self.src_loc
        inv_cov = np.linalg.inv(self.src_cov)
        #Get the boolean map of the cuboid
        x_bounds = (self.src_loc[0]-self.src_widths[0]/2, self.src_loc[0]+self.src_widths[0]/2)
        y_bounds = (self.src_loc[1]-self.src_widths[1]/2, self.src_loc[1]+self.src_widths[1]/2)
        z_bounds = (self.src_loc[2]-self.src_widths[2]/2, self.src_loc[2]+self.src_widths[2]/2)
        
        bool_map = (vox_ctr[:,:,:,0] > x_bounds[0]) & (vox_ctr[:,:,:,0] < x_bounds[1])\
                & (vox_ctr[:,:,:,1] > y_bounds[0]) & (vox_ctr[:,:,:,1] < y_bounds[1])\
                & (vox_ctr[:,:,:,2] > z_bounds[0]) & (vox_ctr[:,:,:,2] < z_bounds[1])
        #Zero out the voxels outside of the cuboid
        self.src_dist[np.logical_not(bool_map)] = 0
        mahalanobis_dist = \
            np.sqrt(
                np.sum(
                    np.einsum(
                        "ijkl, lm -> ijkm", diff, inv_cov) * diff, axis=-1)
                    )
        #Zero out too weak voxel values
        self.src_dist[mahalanobis_dist>3] = 0
        self.src_dist /= self.src_dist.sum()
        #all the src strength are stored in Ci units
        #when calculating the forward projecion we convert it to Bq
        self.src_dist *= self.src_str 

        return self.src_dist

class RingGaussianSource(DistributedSource):
    
    def __init__(self, src_str, src_ctr, r_in, r_out):
        super().__init__(src_str)
        self.src_ctr = src_ctr
        self.r_in = r_in
        self.r_out = r_out

    def get_src_dist(self, vox_ctr):
        r = (self.r_in + self.r_out) / 2
        del_r = (self.r_out - self.r_in) / 2
        self.src_dist = np.zeros(vox_ctr.shape[:-1])
        cov = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,1]]) * del_r**2 / 9
        for i in range(360): 
            theta = np.pi * 2 / 360 * i
            mean = np.array(
                    [self.src_ctr[0] + r * np.cos(theta),
                    self.src_ctr[1] + r * np.sin(theta),
                    self.src_ctr[2]]
                    )
            mult_gauss = multivariate_normal(
                mean = mean,
                cov=cov
                )
            src_dist = mult_gauss.pdf(x=vox_ctr)
            #The following line takes care of 2D voxel space case
            if src_dist.shape != vox_ctr.shape[:-1]:
                src_dist = src_dist.reshape(vox_ctr.shape[:-1])
            diff = vox_ctr-mean
            inv_cov = np.linalg.inv(cov)
            mahalanobis_dist = \
                np.sqrt(
                    np.sum(
                        np.einsum(
                            "ijkl, lm -> ijkm", diff, inv_cov) * diff, axis=-1)
                        )
            #Zero out too weak voxel values
            src_dist[mahalanobis_dist>3] = 0
            self.src_dist += src_dist
            
        self.src_dist /= self.src_dist.sum()
        #all the src strength are stored in Ci units
        #when calculating the forward projecion we convert it to Bq
        self.src_dist *= self.src_str 

        return self.src_dist
    
class UniformRingSource(DistributedSource):
    
    def __init__(self, src_str, src_ctr, r_in, r_out):
        super().__init__(src_str)
        self.src_ctr = src_ctr
        self.r_in = r_in
        self.r_out = r_out

    def get_src_dist(self, vox_ctr):
        r = (self.r_in + self.r_out) / 2
        del_r = (self.r_out - self.r_in) / 2
        self.src_dist = np.zeros(vox_ctr.shape[:-1])
        for i in range(360): 
            theta = np.pi * 2 / 360 * i
            mean = np.array(
                    [self.src_ctr[0] + r * np.cos(theta),
                    self.src_ctr[1] + r * np.sin(theta),
                    self.src_ctr[2]]
                    )
            src_dist = np.where(np.sum((mean.reshape(1, 1, 1, 3) - vox_ctr)**2, axis=-1) < del_r**2, 1, 0)
            self.src_dist += src_dist
        self.src_dist = np.where(self.src_dist > 0, 1.0, 0) 
        #all the src strength are stored in Ci units
        #when calculating the forward projecion we convert it to Bq
        self.src_dist /= self.src_dist.sum()
        self.src_dist *= self.src_str 

        return self.src_dist
    
class UniformGaussianSource(DistributedSource):
    
    def __init__(self, src_str, src_loc, src_cov):
        super().__init__(src_str)
        self.src_cov = src_cov
        self.src_loc = src_loc

    @property
    def src_cov(self):
        return self._src_cov

    @src_cov.setter
    def src_cov(self, cov):
        self._src_cov = cov

    def get_src_dist(self, vox_ctr):
        mult_gauss = multivariate_normal(
            mean = self.src_loc,
            cov=self.src_cov)
        self.src_dist = mult_gauss.pdf(x=vox_ctr)
        #The following line takes care of 2D voxel space case
        if self.src_dist.shape != vox_ctr.shape[:-1]:
            self.src_dist = self.src_dist.reshape(vox_ctr.shape[:-1])
        diff = vox_ctr-self.src_loc
        inv_cov = np.linalg.inv(self.src_cov)
        mahalanobis_dist = \
            np.sqrt(
                np.sum(
                    np.einsum(
                        "ijkl, lm -> ijkm", diff, inv_cov) * diff, axis=-1)
                    )
        #Zero out too weak voxel values
        self.src_dist[mahalanobis_dist>2] = 0
        self.src_dist[mahalanobis_dist<2]=\
            1/np.count_nonzero(mahalanobis_dist<2)
        #all the src strength are stored in Ci units
        #when calculating the forward projecion we convert it to Bq
        self.src_dist *= self.src_str 

        return self.src_dist


class UniformCuboidSource(DistributedSource):
    
    def __init__(self, src_str, src_loc, src_widths):
        """Class for 3D Cuboid shaped, uniformly distributed sources.

        Args:
            src_str (float): Total source strength
            src_loc (array-like): x, y, z coordinate of the center of the rectangular.
            src_widths (array-like): side lengths of the cuboid in x, y, z directions. 
        """
        super().__init__(src_str)
        self.src_widths = src_widths
        self.src_loc = src_loc

    @property
    def src_widths(self):
        return self._src_widths

    @src_widths.setter
    def src_widths(self, src_widths):
        self._src_widths = src_widths

    def get_src_dist(self, vox_ctr):
        #Find the boolean map indicating which voxelsare  within the cuboid.
        x_bounds = (self.src_loc[0]-self.src_widths[0]/2, self.src_loc[0]+self.src_widths[0]/2)
        y_bounds = (self.src_loc[1]-self.src_widths[1]/2, self.src_loc[1]+self.src_widths[1]/2)
        z_bounds = (self.src_loc[2]-self.src_widths[2]/2, self.src_loc[2]+self.src_widths[2]/2)
        
        bool_map = (vox_ctr[:,:,:,0] > x_bounds[0]) & (vox_ctr[:,:,:,0] < x_bounds[1])\
                & (vox_ctr[:,:,:,1] > y_bounds[0]) & (vox_ctr[:,:,:,1] < y_bounds[1])\
                & (vox_ctr[:,:,:,2] > z_bounds[0]) & (vox_ctr[:,:,:,2] < z_bounds[1])
        plt.pcolormesh(bool_map[:,:,0])
        self.src_dist = bool_map * self.src_str / np.count_nonzero(bool_map.ravel())
        return self.src_dist

