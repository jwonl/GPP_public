import numpy as np
from. import random_path
import pathlib

class DetectorPath():
    """
    This is a parent class for all Path classes
    """
    def __init__(self):
        self.path = None
        pass
    
    def save_path(self, path_dir, file_name):
        '''
        Save the path data in .csv extension.
        '''
        assert self.path is not None
        if path_dir[-1] == "/":
            path_dir = path_dir[:-1]
        if file_name[-4:] != ".csv":
            file_name += ".csv"
            
        pathlib.Path(path_dir).mkdir(
            parents=True,
            exist_ok=True)

        np.savetxt(path_dir+'/'+file_name, self.path, )

        return
    
    def load_path(self, path_dir, file_name):
        if path_dir[-1] == "/":
            path_dir = path_dir[:-1]
        if file_name[-4:] != ".csv":
            file_name += ".csv"
        if self.path is not None:
            print("Overwriting the existing path")
        
        self.path = np.loadtxt(path_dir+'/'+file_name, self.path)
        return 
    
        
class WalkingPath(DetectorPath):
    def __init__(self):
        self.path = None
        return
    
    def make_random_path_2D(
        self,
        start_point,
        finish_point,
        path_len,
        d_thres,
        z_coordinate=0,
        extent=None,
        seed=None):
        path = random_path.random_path_2D(
            start_point=start_point,
            finish_point=finish_point,
            path_len=path_len,
            d_thres=d_thres,
            extent=extent,
            seed=seed
        )
        #Make the path 3D by padding zeros
        self.path = np.hstack((path, np.ones((path.shape[0],1))\
            *z_coordinate))
        
        return
        
class RasterPath(DetectorPath):
    def __init__(self):
        self.path = None
        return
    
    def make_raster_path_2D(
        self,
        extent,
        step_size,
        v_steplength,
        z_coordinate = 0,
        starting_point=None,
        h_steplength=None):
        #TODO:implement with the starting point being not none
        
        xmin=extent[0][0]
        xmax=extent[0][1]
        ymin=extent[1][0]
        ymax=extent[1][1]
        
        if h_steplength is None:
            h_steplength = xmax-xmin
            
        if starting_point is None:
            starting_point = (extent[0][0], extent[0][1])
        
        #Number of vertical moves
        n_vmove = (ymax - ymin) // v_steplength + 1 
        n_vmove = int(n_vmove)
        x = np.arange(xmin, xmax, step_size).reshape(-1,1)
        y = np.arange(ymin, ymin + v_steplength, step_size).reshape(-1,1)
        L_path_horizontal = np.hstack([x, np.ones(x.shape) * ymin])
        L_path_vertical = np.hstack([np.ones(y.shape) * xmax, y])
        L_path = np.vstack([L_path_horizontal, L_path_vertical])
        for i in range(n_vmove):
            if i == 0:
                path = np.copy(L_path)  
            else:
                path = np.vstack([path, L_path])
            #Reverse the L Path
            L_path[:,0] = -L_path[:,0] + 2 * (x.max()+x.min()) / 2
            L_path[:,1] += v_steplength
            
        bool_idx = np.where(path[:,1] < ymax)
        path = path[bool_idx]
        #Make the path 3D by padding zeros
        self.path = np.hstack((path, np.ones((path.shape[0],1)) * z_coordinate))
        
        pass
