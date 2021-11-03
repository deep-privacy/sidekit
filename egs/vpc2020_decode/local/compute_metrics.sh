#!/usr/bin/env bash

asv_model="../libri360_train/model/best_test_libri_train_clean_360.pt"
asv_model="../../best_halp_clr_adam_aam0.2_30_b256_vox12.pt_epoch71"

asv_test=()
# librispeech
for suff in 'dev' 'test'; do
  # Baseline on clear speech
  asv_test+=("libri_${suff}_enrolls,libri_${suff}_trials_f")
  asv_test+=("libri_${suff}_enrolls,libri_${suff}_trials_m")

  # Ignorant attacker
  asv_test+=("libri_${suff}_enrolls,libri_${suff}_trials_f_anon")
  asv_test+=("libri_${suff}_enrolls,libri_${suff}_trials_m_anon")

  # Semi/lazy informed attacker (asv not retrained on anon speech)
  asv_test+=("libri_${suff}_enrolls_anon,libri_${suff}_trials_f_anon")
  asv_test+=("libri_${suff}_enrolls_anon,libri_${suff}_trials_m_anon")
done

# vctk diff and common sets
for common in '' '_common'; do
  for suff in 'dev' 'test'; do
    # Baseline on clear speech
    asv_test+=("vctk_${suff}_enrolls,vctk_${suff}_trials_f${common}")
    asv_test+=("vctk_${suff}_enrolls,vctk_${suff}_trials_m${common}")

    # Ignorant attacker
    asv_test+=("vctk_${suff}_enrolls,vctk_${suff}_trials_f${common}_anon")
    asv_test+=("vctk_${suff}_enrolls,vctk_${suff}_trials_m${common}_anon")

    # Semi/lazy informed attacker (asv not retrained on anon speech)
    asv_test+=("vctk_${suff}_enrolls_anon,vctk_${suff}_trials_f${common}_anon")
    asv_test+=("vctk_${suff}_enrolls_anon,vctk_${suff}_trials_m${common}_anon")
  done
done

# Cleanup
for asv_row in "${asv_test[@]}"; do
    while IFS=',' read -r enroll trial; do
        for data_dir in "$enroll" "$trial"; do
          \rm ./data/$data_dir/x_vector.scp || true
        done
    done <<< "$asv_row"
done

for asv_row in "${asv_test[@]}"; do
    while IFS=',' read -r enroll trial; do
        printf 'ASV: %s\n' "$enroll - $trial"

        for data_dir in "$enroll" "$trial"; do
          if [[ ! -f ./data/$data_dir/x_vector.scp ]]; then
            >&2 echo -e "Extracting x-vectors of $data_dir"
            extract_xvectors.py \
              --vad \
              --model $asv_model \
              --wav-scp ./data/$data_dir/wav.scp \
              --out-scp ./data/$data_dir/x_vector.scp || exit 1
          fi
          if [[ ! "$(wc -l < ./data/$data_dir/wav.scp)" -eq "$(wc -l < ./data/$data_dir/x_vector.scp)" ]]; then >&2 echo -e "\nWarning: Something went wrong during the x-vector extraction!\nPlease redo the extraction:\n\trm ./data/$data_dir/x_vector.scp\n" && exit 2; fi
        done

        compute_spk_cosine.py \
          ./data/$trial/trials \
          ./data/$enroll/utt2spk \
          ./data/$trial/x_vector.scp \
          ./data/$enroll/x_vector.scp \
          ./data/$trial/cosine_score_$enroll.txt || exit 1

        compute_metrics.py \
          -k ./data/$trial/trials \
          -s ./data/$trial/cosine_score_$enroll.txt || exit 1

    done <<< "$asv_row"
done
