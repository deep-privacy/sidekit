# LibriSpeech_360

This repository contains the framework for training speaker recognition models on LibriSpeech_360.

### Data preparation

The following script can be used to download and prepare the LibriSpeech dataset for training.

```bash
python ./dataprep.py --save_path data --download
python ./dataprep.py --from data --make-csv
```

In order to use data augmentation, also run:

```bash
python ./dataprep.py --save_path data --augment
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

### Train x-vector (HalfResnet32 w/ aam loss) model

```bash
train.sh
```
