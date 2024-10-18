import numpy as np
import mdtraj as md
import utils
from sklearn.decomposition import PCA
from visualizer import TrajectoryVisualizer

#example
dirs = {
    "adapted":{
        "traj":'../../adapted/production.fit.xtc',
        "top" :'../../adapted/nowater.gro',
        "atom" :'BB'
    },
    "martini":{
        "traj":'../../control/production.fit.xtc',
        "top" :'../../control/nowater.gro',
        "atom" :'BB'

    },
    "allatom":{
        "traj":'../../one-2N4R-PRR/all.fit.1ns.xtc',
        "top" :'../../one-2N4R-PRR/protein.pdb',
        "atom" :'CA'
    }
}


def compute_projection(xyz):
    """
    algorithm for reducing dimentionality

    Parameters:
        xyz: trajectories of shape [frames, length, 3]
    
    Returns:
       result: list(ndarray) of 2D coordinates (shape : [frames, 2])
    """
    length = xyz.shape[1]
    dist_feat = utils.xyz_to_dmap(xyz).reshape(-1,length**2)   
    pca = PCA(n_components=2)  # Adjust n_components as needed
    pca.fit(dist_feat)
    pca_result = pca.transform(dist_feat)
    return pca_result


def main():
    n_samples = 1000 # -1 to load all samples (trajectories)
    ff_type = "allatom"

    xyz, indicies = utils.get_xyz(dirs[ff_type], n_samples=n_samples) # xyz: shape [frames,length, 3]
    projection = compute_projection(xyz) # projection : (shape : [frames, 2])
    
    visualizer = TrajectoryVisualizer(projection, xyz, ff_type)
    visualizer.show()

main()
