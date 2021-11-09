#!/bin/bash

#################################################################################################
# Script to launch multiple experiments template
# First it creates launch training scripts and configuration files
# Then it launch all training scripts created
#
# This version is written for running on Grid5000 with OAR.
# To change this configuration, change the launching experiments part at the bottom of this file
##################################################################################################

# Parameters of experiments to run
model_type="scratch transfer" # Specify if model is trained from scratch or with transfer learning
aug_type="aug no_aug" # Specify if data augmentation is required
dataset_list="libri_anon|wav libri_vad|flac" # Fill with dataset name|audio file extension. Dataset_name is the name of csv file in list directory
out_dir="tmp" # Directory to store train scripts and config files for all experiments
transfer_model="../../best_halp_clr_adam_aam0.2_30_b256_vox12.pt_epoch71" # Path to the pretrained model. used when model_type = transfer


# Prepare output directories
cfg_dir=$out_dir/cfg
mkdir -p $out_dir
mkdir -p $cfg_dir

# Prepare scripts and config files
for dataset in $dataset_list
do
  for model in $model_type
  do
    for aug in $aug_type
    do
      dataset_name=$(echo $dataset | cut -d '|' -f1 )
      extension=.$(echo $dataset | cut -d '|' -f2)
      suffix="$dataset_name"_"$model"_"$aug"
      new_train_sh=$out_dir/train_$suffix.sh
      cp train.sh $new_train_sh
      new_dataset_yaml=$cfg_dir/dataset_$suffix.yaml
      cp cfg/Librispeech.yaml $new_dataset_yaml
      new_model_yaml=$cfg_dir/model_$suffix.yaml
      cp cfg/model.yaml $new_model_yaml
      new_training_yaml=$cfg_dir/training_$suffix.yaml
      cp cfg/training.yaml $new_training_yaml

      # Modifying train.sh script
      sed -i "s|dataset_file=.*|dataset_file=$new_dataset_yaml|g" $new_train_sh
      sed -i "s|model_file=.*|model_file=$new_model_yaml|g" $new_train_sh
      sed -i "s|training_file=.*|training_file=$new_training_yaml|g" $new_train_sh

      # Modifying dataset.yaml
      sed -i "s|dataset_csv:.*|dataset_csv: list/$dataset_name.csv|g" $new_dataset_yaml

      sed -i "s|data_file_extension:.*|data_file_extension: $extension|g" $new_dataset_yaml
      if [ $aug = "aug" ]; then
        sed -i "0,/pipeline:/{s|pipeline:.*|pipeline: add_reverb,add_noise,filtering,phone_filtering,codec|g}" $new_dataset_yaml
      elif [ $aug = "no_aug" ]; then
        sed -i "0,/pipeline:/{s|pipeline:.*|pipeline: filtering,phone_filtering,codec|g}" $new_dataset_yaml
      fi

      # Modifying model.yaml
      if [ $model = "scratch" ]; then
        sed -i "s|initial_model_name:.*|initial_model_name:|g" $new_model_yaml
        sed -i "s|reset_parts:.*|reset_parts: []|g" $new_model_yaml
      elif [ $model = "transfer" ]; then
        sed -i "s|initial_model_name:.*|initial_model_name: $transfer_model|g" $new_model_yaml
        sed -i "s|reset_parts:.*|reset_parts: [after_speaker_embedding]|g" $new_model_yaml
      fi

      # Modifying training.yaml
      sed -i "s|log_file:.*|log_file: log/$suffix.log|g" $new_training_yaml
      sed -i "s|tmp_model_name:.*|tmp_model_name: model/tmp_$suffix.pt|g" $new_training_yaml
      sed -i "s|best_model_name:.*|best_model_name: model/best_$suffix.pt|g" $new_training_yaml
    done
  done
done

# Launch experiments on Grid5000
oar_out_dir="oarOutDir/"
mkdir -p $oar_out_dir
for file in tmp/*.sh; do
  chmod 744 $file
  curFilename=$(basename $file)
  oarsub -q production -p "cluster <> 'grimani' and cluster <> 'graphique' and cluster <> 'gruss'" -l /host=1/gpu=2,walltime="24:00:00" \
             --stdout="$oar_out_dir/$curFilename.%jobid%.output" --stderr="$oar_out_dir/$curFilename.%jobid%.error" \
             "$file" | grep "OAR_JOB_ID"
done
