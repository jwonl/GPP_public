import numpy as np
from .source import *
from .utilities import *
from pathlib import Path
from tqdm import tqdm


class Scenario:
    CI_TO_BQ = 3.7e10

    def __init__(self, path, srcs, dets, extent, vox_size, int_time, bkg=0, seed=12345):
        self.path_cls = path
        self.path = path.path
        self.extent = extent
        self.srcs = srcs
        self.dets = dets
        self.int_time = int_time
        self.bkg = bkg  # constant background cps
        self.vox_size = vox_size
        self.x_ctr = np.arange(extent[0, 0], extent[0, 1], vox_size) + vox_size / 2
        self.y_ctr = np.arange(extent[1, 0], extent[1, 1], vox_size) + vox_size / 2
        self.z_ctr = np.arange(extent[2, 0], extent[2, 1], vox_size) + vox_size / 2
        self.x_bound = np.arange(extent[0, 0], extent[0, 1] + vox_size, vox_size)
        self.y_bound = np.arange(extent[1, 0], extent[1, 1] + vox_size, vox_size)
        self.z_bound = np.arange(extent[2, 0], extent[2, 1] + vox_size, vox_size)
        self.Y_bound, self.X_bound, self.Z_bound = np.mgrid[
            extent[1, 0] : extent[1, 1] + vox_size : vox_size,
            extent[0, 0] : extent[0, 1] + vox_size : vox_size,
            extent[2, 0] : extent[2, 1] + vox_size : vox_size,
        ]
        self.vox_bound = np.stack((self.X_bound, self.Y_bound, self.Z_bound), axis=-1)

        self.Y_ctr, self.X_ctr, self.Z_ctr = np.mgrid[
            extent[1, 0] + vox_size / 2 : extent[1, 1] : vox_size,
            extent[0, 0] + vox_size / 2 : extent[0, 1] : vox_size,
            extent[2, 0] + vox_size / 2 : extent[2, 1] : vox_size,
        ]
        self.vox_ctr = np.stack((self.X_ctr, self.Y_ctr, self.Z_ctr), axis=-1)
        self.dims = self.vox_ctr.shape[:-1]
        self.src_dist = np.zeros(self.X_ctr.shape)
        self.n_pose = self.path.shape[0]
        self.n_vox = np.size(self.X_ctr)
        self.sys_mat = None
        np.random.seed(seed)
        self.get_src_dist()
        self.forward_proj()

    def get_src_dist(self):
        for s in self.srcs:
            self.src_dist += s.get_src_dist(self.vox_ctr)

    def forward_proj(self):
        self.fp_counts = np.zeros((self.path.shape[0], len(self.dets)))
        # Dictionary to keep track of the contribution from each source and background component
        self.fp_counts_dict = {}
        for d_num, d in enumerate(self.dets):
            # The following lines calculate system matrix
            # TODO it'd be better to have a seperate class method
            # dedicated to calculate system matrix.
            r = np.linalg.norm(
                (
                    self.vox_ctr[np.newaxis, :, :, :, :]
                    - self.path[:, np.newaxis, np.newaxis, np.newaxis, :]
                ),
                axis=-1,
            )
            geo_eff = 0.5 * (1 - r / np.sqrt(d.det_rad**2 + r**2))
            del r
            self.sys_mat = np.reshape(
                geo_eff * d.abs_eff * self.int_time, (self.n_pose, -1)
            )
            del geo_eff

            for s_num, s in tqdm(enumerate(self.srcs)):      
                cnts = (self.sys_mat @ s.src_dist.reshape(-1, 1)) * self.CI_TO_BQ
                self.fp_counts_dict[d_num, s_num] = cnts.ravel()
                self.fp_counts[:, d_num] += cnts.ravel()
        bkg_counts = self.int_time * self.bkg
        self.fp_counts_dict["bkg"] = bkg_counts
        self.fp_counts += bkg_counts
        self.counts = np.random.poisson(self.fp_counts)

    def draw_img(self, saveimg=False, fname=None, **kwargs):
        return draw_2D_img(self, saveimg=saveimg, fname=fname, **kwargs)
    
    def draw_forward_proj(self, saveimg=False, fname=None, **kwargs):
        if "grid" in kwargs.keys():
                grid = kwargs.pop("grid")    
        else:
            grid = True
        count_data = [
            {
                "counts": self.counts.sum(axis=1),
                "kwargs": {"label": "Measured counts", **kwargs},
                "uncertainty": None,
            }
        ]

        plot_count_data(
            count_data=count_data, int_time=self.int_time, saveimg=saveimg, fname=fname, grid=grid
        )
        pass

    def save_counts(self, dir, fname):
        """
        Save the foward projected count data in a simple text file
        """
        assert self.counts is not None

        if dir[-1] == "/":
            dir = dir[:-1]
        if fname[-4:] != ".csv":
            fname += ".csv"

        Path(dir).mkdir(parents=True, exist_ok=True)

        np.savetxt(dir + "/" + fname, self.counts)
        return

    def save_scanario(self):
        # TO BE IMPLEMENTED
        return

    def load_scanario(self):
        # TO BE IMPLEMENTED
        return
