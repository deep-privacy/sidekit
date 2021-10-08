SIDEKIT
=======

SIDEKIT is an open source package for Speaker and Language recognition.  
This repo provides the python/pytorch implementation the Deep Neural Network based Automatic Speaker Recognition systems and training losses of: [Larcher/sidekit](https://git-lium.univ-lemans.fr/Larcher/sidekit)

Authors: Anthony Larcher & Kong Aik Lee & Sylvain Meignier

## Installation

```sh
git clone https://github.com/deep-privacy/sidekit
cd sidekit
# you might need to adjust $CUDAROOT of ./install.sh
# to match your cuda config. default /usr/local/cuda
./install.sh
```

## Usage

#### For kaldi-like wav.scp

Model: HalfResNet34  
Trained with Loss: [Large margin arc distance](https://arxiv.org/pdf/1904.03479.pdf)  
Trained on: VoxCeleb1 & 2  
Test EER on vox1-O: `1.20 %`  

```sh
# activate the miniconda venv
. ./env.sh

# download trained model
wget https://github.com/deep-privacy/sidekit/releases/download/sidekit_v0.1/best_halp_clr_adam_aam0.2_30_b256_vox12.pt_epoch71

# wav.scp to extract (kaldi-like)
cat ./wav_example.scp

lbi-1272-128104-0000 flac -c -d -s ex/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac |
lbi-1272-128104-0001 flac -c -d -s ex/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac |
lbi-1272-128104-0002 flac -c -d -s ex/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac |

# extract and store the x-vectors in a scp,ark file
python3 sidekit/local/extract_xvectors.py --model ./best_halp_clr_adam_aam0.2_30_b256_vox12.pt_epoch71 \
  --device cuda --wav-scp ./wav_example.scp --out-scp ./x-vector.scp
```

#### For Python

```python
import torch
import torchaudio
from sidekit.nnet.xvector import Xtractor

model_path = "./best_halp_clr_adam_aam0.2_30_b256_vox12.pt_epoch71"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_config = torch.load(model_path, map_location=device)

model_opts = model_config["model_archi"]
if "embedding_size" not in model_opts:
    model_opts["embedding_size"] = 256
xtractor = Xtractor(model_config["speaker_number"],
                 model_archi=model_opts["model_type"],
                 loss=model_opts["loss"]["type"],
                 embedding_size=model_opts["embedding_size"])

xtractor.load_state_dict(model_config["model_state_dict"], strict=True)
xtractor = xtractor.to(device)
xtractor.eval()

wav_tensor, sample_rate = torchaudio.load("example.wav")
_, vec = xtractor(wav_tensor.to(device), is_eval=True)
print(vec.shape)
```

---

Model training, corpus download, and corpus preparation tutorial will come later.
