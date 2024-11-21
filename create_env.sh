#!/bin/bash

# 1. Create Virtual Env
# conda create -n flame2smplx python=3.8
# conda activate flame2smplx

# 2. Install PyTorch
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
python setup.py install

# Install mesh
conda install -c anaconda boost
git clone https://github.com/MPI-IS/mesh
pip install --upgrade -r mesh/requirements.txt
pip install --no-deps --verbose --no-cache-dir mesh/.
# pip install --no-deps --install-option="--boost-location=$$BOOST_INCLUDE_DIRS" --verbose --no-cache-dir mesh/.

# Install torch-trust-ncg
git clone https://github.com/vchoutas/torch-trust-ncg.git
cd torch-trust-ncg/
python setup.py install
cd ..

pip install open3d loguru omegaconf trimesh chumpy
pip install numpy==1.23.5