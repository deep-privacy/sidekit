# VoxCeleb12

This repository contains the framework for training speaker recognition models on VoxCeleb12.

### Data preparation (from: https://github.com/clovaai/voxceleb_trainer)

The following script can be used to download and prepare the VoxCeleb dataset for training.

```bash
python ./dataprep.py --save_path data --download --user USERNAME --password PASSWORD
python ./dataprep.py --save_path data --extract
python ./dataprep.py --save_path data --convert
```
In order to use data augmentation, also run:

```bash
python ./dataprep.py --save_path data --augment
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.
