#!/bin/bash

set -e

nj=$(nproc)

home=$PWD

# python/CONDA
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh

# PYTORCH version
torch_version=1.8.2
torchvision_version=0.9.2
torchaudio_version=0.8.2
torch_wheels="https://download.pytorch.org/whl/lts/1.8/torch_lts.html"

# CUDA version
cuda_version=10.2


venv_dir=$PWD/venv

mark=.done-venv
if [ ! -f $mark ]; then
  echo " == Making python virtual environment =="
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && rm -r $venv_dir
  sh $name -b -p $venv_dir || exit 1
  . $venv_dir/bin/activate

  echo "Installing conda dependencies"
  yes | conda install -c conda-forge sox || exit 1
  yes | conda install -c conda-forge libflac || exit 1
  yes | conda install -c conda-forge cudatoolkit=$cuda_version || exit 1
  touch $mark
fi
source $venv_dir/bin/activate

exit 0

# CUDA version
CUDAROOT=/usr/local/cuda
if [ "$(id -g --name)" == "lium" ]; then
  CUDAROOT=/opt/cuda/10.2 # LIUM Cluster
  echo "Using local \$CUDAROOT: $CUDAROOT"
fi
_cuda_version=$($CUDAROOT/bin/nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
if [[ $cuda_version != $_cuda_version ]]; then
  echo "CUDA env not properly setup! (installed cuda v$cuda_version != in path cuda v$_cuda_version)"
  exit 1
fi
cuda_version_witout_dot=$(echo $cuda_version | xargs | sed 's/\.//')

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDAROOT/include $CFLAGS"
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
echo "if [ \$(which python) != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi; export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" > env.sh

mark=.done-pytorch
if [ ! -f $mark ]; then
  echo " == Installing pytorch $torch_version for cuda $cuda_version =="
  pip3 install torch==$torch_version+cu$cuda_version_witout_dot torchvision==$torchvision_version+cu$cuda_version_witout_dot torchaudio==$torchaudio_version -f $torch_wheels
  cd $home
  touch $mark
fi

mark=.done-python-requirements
if [ ! -f $mark ]; then
  echo " == Installing python libraries =="

  # sidekit additional req
  pip3 install matplotlib==3.4.3
  pip3 install SoundFile==0.10.3.post1
  pip3 install PyYAML==5.4.1
  pip3 install h5py==3.2.1
  pip3 install ipython==7.27.0

  cd $home
  touch $mark
fi

mark=.done-sidekit
if [ ! -f $mark ]; then
  echo " == Building sidekit =="
  cd sidekit
  pip3 install -e .
  cd $home
  touch $mark
fi

echo " == Everything got installed successfully =="
