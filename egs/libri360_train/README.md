# LibriSpeech_360

This repository contains the framework for training speaker recognition models on LibriSpeech_360.

### Data preparation

The following script can be used to download and prepare the LibriSpeech dataset for training.

```bash
python ./dataprep.py --save-path data --download
python ./dataprep.py --from ./data --make-train-csv # set --filter-dir if your data dir structure differ from the '--download' one
```

In order to use data augmentation, also run:

```bash
python ./dataprep.py --save-path data --download-augment
python ./dataprep.py --from ./data/RIRS_NOISES --make-csv-augment-reverb
python ./dataprep.py --from ./data/musan_split --make-csv-augment-noise
```

In addition to the Python dependencies, `wget` must be installed on the system.

### Train x-vector (HalfResnet32 w/ aam loss) model

```bash
# Remove non speech
apply_vad_on_csv.py --nj 64 --in-csv ./list/libri.csv --out-csv ./list/libri_vad.csv --out-audio-dir ./data/libri_vad --extension-name flac
# Train the sidekit model
train.sh
```
