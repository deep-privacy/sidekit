#!/bin/bash

# Parameters of experiments to run
model_list="../libri360_train/model/best_libri_vad_transfer_no_aug.pt_epoch103_EER_0.04_ACC_100.00.pt|Libri_vad_transfer_no_aug \
            ../libri360_train/model/best_libri_vad_transfer_aug.pt_epoch51_EER_0.00_ACC_99.95.pt|Libri_vad_transfer_aug \
            ../libri360_train/model/best_libri_vad_scratch_no_aug.pt_epoch140_EER_0.07_ACC_100.00.pt|Libri_vad_scratch_no_aug \
            ../libri360_train/model/best_libri_vad_scratch_aug.pt_epoch138_EER_0.18_ACC_99.95.pt|Libri_vad_scratch_aug \
            ../libri360_train/model/best_libri_anon_transfer_no_aug.pt_epoch28_EER_7.11_ACC_72.22.pt|Libri_anon_spk_transfer_no_aug \
            ../libri360_train/model/best_libri_anon_transfer_aug.pt_epoch109_EER_7.66_ACC_80.33.pt|Libri_anon_spk_transfer_aug \
            ../libri360_train/model/best_libri_anon_scratch_no_aug.pt_epoch94_EER_6.14_ACC_80.88.pt|Libri_anon_spk_scratch_no_aug \
            ../libri360_train/model/best_libri_anon_scratch_aug.pt_epoch125_EER_5.97_ACC_81.76.pt|Libri_anon_spk_scratch_aug"

# Prepare output directories
local_dir="local"
mkdir -p $local_dir
results_out_dir="resultsOutDir/"
log_dir="oarOutDir/"
mkdir -p $log_dir
mkdir -p $results_out_dir

# Prepare scripts and config files
for model in $model_list
do
  model_path=$(echo $model | cut -d '|' -f1 )
  model_name=$(echo $model | cut -d '|' -f2)
  new_compute_metrics=$local_dir/compute_metrics_$model_name.sh
  cp $local_dir/compute_metrics.sh $new_compute_metrics
  chmod 744 $new_compute_metrics
  sed -i "s|asv_model=.*|asv_model=$model_path|g" $new_compute_metrics
  # Launching compute_metrics with OAR. Change this to fit to your cluster requirements
  oarsub -q production -p "cluster <> 'grimani' and cluster <> 'graphique' and cluster <> 'gruss'" -l /host=1/gpu=2,walltime="24:00:00" \
             --stdout="$log_dir/$model_name.%jobid%.output" --stderr="$log_dir/$model_name.%jobid%.error" \
             "source ../../env.sh && $new_compute_metrics" | grep "OAR_JOB_ID"
done