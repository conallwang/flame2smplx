import scipy
import numpy as np
import pickle

import scipy.sparse

# Download from https://smpl-x.is.tue.mpg.de/download.php
# MANO and FLAME vertex indices
correspondence_path = "/path/to/flame2smplx/transfer_data/SMPL-X__FLAME_vertex_ids.npy"
corr = np.load(correspondence_path)  # 5023

smplx_num_vertices = 10475
flame_num_vertices = corr.shape[0]

matrix = np.zeros((smplx_num_vertices, flame_num_vertices))
# flame2smplx_deftrafo = {"matrix": np.zeros((smplx_num_vertices, flame_num_vertices))}

idcs = list(range(flame_num_vertices))
matrix[corr, idcs] = 1
# flame2smplx_deftrafo["matrix"][corr, idcs] = 1
flame2smplx_deftrafo = {"mtx": scipy.sparse.coo_matrix(matrix)}

with open("/path/to/flame2smplx/transfer_data/flame2smplx_deftrafo.pkl", "wb") as handle:
    pickle.dump(flame2smplx_deftrafo, handle)
