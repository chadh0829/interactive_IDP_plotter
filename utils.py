import pandas as pd
import mdtraj as md
import re 
import numpy as np
from typing import Dict, Any, Tuple

def parse_xvg(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    column_names = ["Time (ps)"]  # The first column is always "Time (ps)"
    units = None
    data = []
    idx = 0
    
    for line in lines:
        # Skip comment lines
        if line.startswith('#'):
            continue
        
        # Extract units from the y-axis label line
        if line.startswith('@    yaxis  label'):
            # Extract the units part from the label line (between parentheses)
            units = re.findall(r"\(.*?\)", line)

        # Extract column names from lines starting with '@ s'
        if line.startswith('@ s'):
            label = line.split('"')[1]  # Extract text between quotes as the label
            # Append the units information if available
            if units:
                column_names.append(f"{label} ({units[idx]})")
                idx += 1
            else:
                column_names.append(label)
        
        # Process lines with numeric data
        if not (line.startswith('@') or line.startswith('#')):
            split_line = line.strip().split()
            if split_line:
                data.append([float(value) for value in split_line])

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data, columns=column_names)
    
    return df

def xyz_to_dmap(xyz):
    """
    Gets an ensemble of xyz conformations with shape (N, L, 3) and
    returns the corresponding distance matrices with shape (N, L, L).
    """ 
    return np.sqrt(np.sum(np.square(xyz[:,None,:,:]-xyz[:,:,None,:]), axis=3))

def dmap_to_cmap(dmap, threshold=0.8, pseudo_count=0.01):
    """
    Gets a trajectory of distance maps with shape (N, L, L) and
    returns a (L, L) contact probability map.
    """
    n = dmap.shape[0]
    cmap = ((dmap <= threshold).astype(int).sum(axis=0)+pseudo_count)/n
    return cmap

def get_topol(dir) -> md.Topology:
    t = md.load(dir["traj"],top = dir["top"])
    ca = [a.index for a in t.topology.atoms if a.name == dir["atom"]]
    t.restrict_atoms(ca)
    return t


def get_xyz(
        dir: Dict[str,str], 
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    load trajectory from directory as xyz coordinate
    Parameters:
        dir: Dictionary containing directories of a trajectory, topology and a representative atom type for each residue (i.e. CA)
        n_samples: number of samples to load, -1 means to load all trajectories
    Returns:
        xyz: xyz coordinate with shape [frames, length, 3]
        indicies: ndarray index of samples
    """
    t = get_topol(dir)
    xyz = t.xyz
    if n_samples == -1:
        return xyz.copy(), np.arange(xyz.shape[0])
    else:
        indices = np.random.choice(xyz.shape[0], n_samples, replace=False)
        return xyz[indices].copy(), indices
    
def KabschAlign(p,q):
    p = p.reshape(-1,3)
    q = q.reshape(-1,3)

    q -= q.mean(axis=0)
    p -= p.mean(axis=0)

    H = p.T@q
    U, S, Vh = np.linalg.svd(H,full_matrices=True)
    d = 1 if np.linalg.det(U@Vh) > 0 else -1
    D = np.eye(3,3)
    D[2][2] = d
    R = U @ D @ Vh
    print()

    qbar = q @ R.T
    # qbar = q

    return qbar, np.sqrt(np.average((p-qbar)**2))
def compute_rg(xyz):
    """
    Adapted from the mdtraj library: https://github.com/mdtraj/mdtraj.
    """
    num_atoms = xyz.shape[1]
    masses = np.ones(num_atoms)
    weights = masses / masses.sum() 
    mu = xyz.mean(1)
    centered = (xyz.transpose((1, 0, 2)) - mu).transpose((1, 0, 2))
    squared_dists = (centered ** 2).sum(2)
    Rg = (squared_dists * weights).sum(1) ** 0.5
    return Rg