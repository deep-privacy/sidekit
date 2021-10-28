# VoxCeleb12

This repository contains the framework for training speaker recognition models on VoxCeleb12.

### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training.

```bash
python ./dataprep.py --save-path data --download --user USERNAME --password PASSWORD
python ./dataprep.py --save-path ./data --convert
python ./dataprep.py --from ./data --make-train-csv # set --filter-dir if your data dir structure differ from the '--download' one (i.e.: voxceleb1/dev/wav/,voxceleb2/dev/wav/)
```
In order to use data augmentation, also run:

```bash
python ./dataprep.py --save-path data --download-augment
python ./dataprep.py --from ./data/RIRS_NOISES --make-csv-augment-reverb
python ./dataprep.py --from ./data/musan_split --make-csv-augment-noise
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.
