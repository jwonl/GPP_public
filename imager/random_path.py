"""
Created on Sun Jan 26 14:29:57 2020

@author: jaewon
"""
import numpy as np

def random_path_2D(
    start_point,
    finish_point,
    path_len,
    d_thres,
    extent=None,
    seed=None):
    
    if extent is not None:
        assert extent.shape == (3,2)

    if seed is not None:
        np.random.seed(seed)
        
    d = np.linalg.norm(start_point-finish_point)
    diff = finish_point-start_point
    center = (finish_point+start_point)/2
    theta = np.arctan2(diff[1],diff[0])
    c = d/2
    a = path_len/2
    b = np.sqrt(a**2-c**2)
    
    while True:
        r = np.sqrt(np.random.rand())
        phi = np.random.uniform(0,2*np.pi)
        elong = np.array([a*r*np.cos(phi),b*r*np.sin(phi)])
        rot_mat = np.array(
            [[np.cos(theta),-np.sin(theta)],
            [np.sin(theta),np.cos(theta)]])
        sampled_point = np.matmul(rot_mat,elong) + center
        if extent is not None:
            boundary_check = np.logical_and(
                (sampled_point.reshape(-1,1) - extent[:-1,0]) > 0,
                (extent[:-1,1] - sampled_point.reshape(-1,1)) > 0)
            if np.all(boundary_check):
                break
            else:
                continue
        elif extent is None:
            break

    start_to_cent = np.linalg.norm(sampled_point-start_point)
    finish_to_cent = np.linalg.norm(sampled_point-finish_point)
    left_d = path_len * (start_to_cent/(start_to_cent+finish_to_cent))
    right_d = path_len * (finish_to_cent/(start_to_cent+finish_to_cent))
    if left_d<d_thres and right_d<d_thres:
        return np.vstack([start_point,sampled_point,finish_point])
    elif left_d>d_thres and right_d<d_thres:
        return np.vstack([random_path_2D(start_point,sampled_point,left_d,d_thres, extent),finish_point])
    elif left_d<d_thres and right_d>d_thres:
        return np.vstack([start_point,random_path_2D(sampled_point,finish_point,right_d,d_thres, extent)])
    elif left_d>d_thres and right_d>d_thres:
        return np.vstack([random_path_2D(start_point,sampled_point,left_d,d_thres, extent),
        random_path_2D(sampled_point,finish_point,right_d,d_thres, extent)[1:,:]])

'''
for i in range(0,10):
    fig, ax = plt.subplots(1,1)
    s = np.array([-10,-10])
    f = np.array([10,10])

    a = random_path_2D(s,f,30,0.2)
    a = np.hstack((a,np.zeros((np.size(a,0),1))))

    plt.scatter(a[:,0],a[:,1],s=0.4, color = 'r')
    np.savetxt('pose/random_pose('+str(i)+').csv', a, delimiter = ',')
    b = np.linalg.norm(a[1:]-a[:-1],axis=1)
    print(np.mean(b))
'''
    