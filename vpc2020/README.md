VoicePrivacy Challenge 2020 SIDEKIT evaluation
==============================================

This repository provides the SIDEKIT-based ASV evaluation of the [VPC2020](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020).

#### Prepare VPC2020 clear and anonymized data

```bash
# Download VPC2020 clear/anonymized speech
cp /home/pchampion/vpc_baseline_speech/vpc2020_clear_anon_speech.zip . # TODO upload it somewhere
unzip -P XXXX vpc2020_clear_anon_speech.zip

# make data dir
local/create_data_dir.sh
```

#### Extract x-vectors and compute metrics
```bash
local/compute_metrics.sh > results.txt
```
