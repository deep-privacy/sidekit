VoicePrivacy Challenge 2020 SIDEKIT evaluation
==============================================

This repository provides the SIDEKIT-based ASV evaluation of the [VPC2020](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020).

#### Prepare VPC2020 clear and anonymized data

```bash
# Download VPC2020 clear/anonymized speech
wget https://huggingface.co/datasets/Champion/vpc2020_clear_anon_speech/resolve/main/vpc2020_clear_anon_speech.zip
unzip -P XXXX vpc2020_clear_anon_speech.zip

# make data dir
local/create_data_dir.sh
```

#### Extract x-vectors and compute metrics
```bash
local/compute_metrics.sh > results.txt
```

#### Train x-vectors extractor from Librispeech
```bash
# Create dataset csv file for model input (make sure you have download Librispeech dataset before)
python3 local/create_dataset_csv.py --libri-root /home/LibriSpeechRoot/ --out_filepath list/libri360.csv

# Change YAML files in cfg directory to personalize training (hyperparameters, ...)

# Run training
local/train.sh 
```