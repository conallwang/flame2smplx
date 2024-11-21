# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

from omegaconf import OmegaConf
from dataclasses import dataclass


@dataclass
class MeshFolder:
    data_folder: str = "data/meshes"
    ext: str = ".obj"


@dataclass
class DatasetConfig:
    num_workers: int = 0
    name: str = "mesh-folder"
    mesh_folder: MeshFolder = MeshFolder()


conf = OmegaConf.structured(DatasetConfig)
