#!/bin/bash

set -e

home=$PWD

# CUDA version
CUDAROOT=/usr/local/cuda

# Define LOCAL CUDA version here:

# LIUM Cluster
if [ "$(id -g --name)" == "lium" ]; then
  CUDAROOT=/opt/cuda/10.2
  echo "Using local \$CUDAROOT: $CUDAROOT"
fi

if [ ! -d $CUDAROOT ]; then
  echo "CUDAROOT: '$FILE' does not exist."
  echo "Installing for CPU compute platform!"
  compute_platform="cpu"
  compute_platform_witout_dot=$compute_platform
  sleep 2
else
  compute_platform=$($CUDAROOT/bin/nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
  compute_platform_witout_dot=$(echo cu"$compute_platform" | xargs | sed 's/\.//')
fi

# CONDA
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh

# PYTORCH
torch_version=1.8.2
torchvision_version=0.9.2
torchaudio_version=0.8.2

torch_wheels="https://download.pytorch.org/whl/lts/1.8/torch_lts.html"

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
  yes | conda install -c conda-forge sox
  yes | conda install -c conda-forge libflac
  touch $mark
fi
source $venv_dir/bin/activate

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDAROOT/include $CFLAGS"
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

echo "if [ \"\$(which python)\" != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi; export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" > env.sh

mark=.done-pytorch
if [ ! -f $mark ]; then
  echo " == Installing pytorch $torch_version for cuda $compute_platform =="
  # pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  pip3 install torch==$torch_version+$compute_platform_witout_dot torchvision==$torchvision_version+$compute_platform_witout_dot torchaudio==$torchaudio_version -f $torch_wheels
  cd $home
  touch $mark
fi

mark=.done-sidekit
if [ ! -f $mark ]; then
  echo " == Building sidekit =="
  pip3 install -e .
  cd $home
  touch $mark
fi


mark=.done-other-python-requirements
if [ ! -f $mark ]; then
  echo " == Installing python libraries =="

  pip3 install kaldiio
  pip3 install tabulate

  cd $home
  touch $mark
fi

mark=.done-anonymization_metrics
if [ ! -f $mark ]; then
  cd tools
  git clone https://gitlab.inria.fr/magnet/anonymization_metrics.git
  cd $home
  pip3 install seaborn
  touch $mark
fi

echo " == Everything got installed successfully =="
