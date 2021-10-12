#!/usr/bin/env bash

home=$PWD

mkdir -p data

for name in wav_anon wav_clear; do
      [ ! -d $name ] && echo "Directory $name does not exist.\nPlease download and unzip vpc_baseline_speech.zip" && exit 1
done

search_dir="./wav_anon"
for data_dir in "$search_dir"/*; do
  wav_path=${data_dir}/nsf_output_wav
  echo $wav_path

  out_data=./data/$(basename ${data_dir}_anon)
  out_data=$(realpath $out_data)
  mkdir -p $out_data
  ls $wav_path | cut -d. -f1 | awk -v p="$wav_path" '{print $1, "sox", p"/"$1".wav", "-t wav -R -b 16 - |"}' > $out_data/wav.scp
done


search_dir="./wav_clear"
temp=$(mktemp)
for suff in dev test; do

  dset=$search_dir/libri_$suff
  utils/subset_data_dir.sh --utt-list $dset/enrolls $dset ./data/$(basename ${dset}_enrolls) || exit 1
  cp $dset/enrolls ./data/$(basename ${dset}_enrolls)/enrolls || exit 1

  cut -d' ' -f2 $dset/trials_f | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ./data/$(basename ${dset}_trials_f) || exit 1
  cp $dset/trials_f ./data/$(basename ${dset}_trials_f)/trials || exit 1

  cut -d' ' -f2 $dset/trials_m | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ./data/$(basename ${dset}_trials_m) || exit 1
  cp $dset/trials_m ./data/$(basename ${dset}_trials_m)/trials || exit 1

  utils/combine_data.sh ./data/$(basename ${dset}_trials_all) ./data/$(basename ${dset}_trials_f) ./data/$(basename ${dset}_trials_m) || exit 1
  cat ./data/$(basename ${dset}_trials_f)/trials ./data/$(basename ${dset}_trials_m)/trials > ./data/$(basename ${dset}_trials_all)/trials

  cp ./data/$(basename ${dset}_trials_f)/utt2spk ./data/$(basename ${dset}_trials_f_anon)
  cp ./data/$(basename ${dset}_trials_m)/utt2spk ./data/$(basename ${dset}_trials_m_anon)

  cp ./data/$(basename ${dset}_trials_f)/trials ./data/$(basename ${dset}_trials_f_anon)
  cp ./data/$(basename ${dset}_trials_m)/trials ./data/$(basename ${dset}_trials_m_anon)

  cp ./data/$(basename ${dset}_enrolls)/utt2spk ./data/$(basename ${dset}_enrolls_anon)
  cp ./data/$(basename ${dset}_enrolls)/enrolls ./data/$(basename ${dset}_enrolls_anon)

  dset=$search_dir/vctk_$suff
  utils/subset_data_dir.sh --utt-list $dset/enrolls_mic2 $dset ./data/$(basename ${dset}_enrolls) || exit 1
  cp $dset/enrolls_mic2 ./data/$(basename ${dset}_enrolls)/enrolls || exit 1

  cut -d' ' -f2 $dset/trials_f_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ./data/$(basename ${dset}_trials_f) || exit 1
  cp $dset/trials_f_mic2 ./data/$(basename ${dset}_trials_f)/trials || exit 1

  cut -d' ' -f2 $dset/trials_f_common_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ./data/$(basename ${dset}_trials_f_common) || exit 1
  cp $dset/trials_f_common_mic2 ./data/$(basename ${dset}_trials_f_common)/trials || exit 1

  utils/combine_data.sh ./data/$(basename ${dset}_trials_f_all) ./data/$(basename ${dset}_trials_f) ./data/$(basename ${dset}_trials_f_common) || exit 1
  cat ./data/$(basename ${dset}_trials_f)/trials ./data/$(basename ${dset}_trials_f_common)/trials > ./data/$(basename ${dset}_trials_f_all)/trials

  cut -d' ' -f2 $dset/trials_m_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ./data/$(basename ${dset}_trials_m) || exit 1
  cp $dset/trials_m_mic2 ./data/$(basename ${dset}_trials_m)/trials || exit 1

  cut -d' ' -f2 $dset/trials_m_common_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp $dset ./data/$(basename ${dset}_trials_m_common) || exit 1
  cp $dset/trials_m_common_mic2 ./data/$(basename ${dset}_trials_m_common)/trials || exit 1

  utils/combine_data.sh ./data/$(basename ${dset}_trials_m_all) ./data/$(basename ${dset}_trials_m) ./data/$(basename ${dset}_trials_m_common) || exit 1
  cat ./data/$(basename ${dset}_trials_m)/trials ./data/$(basename ${dset}_trials_m_common)/trials > ./data/$(basename ${dset}_trials_m_all)/trials

  utils/combine_data.sh ./data/$(basename ${dset}_trials_all) ./data/$(basename ${dset}_trials_f_all) ./data/$(basename ${dset}_trials_m_all) || exit 1
  cat ./data/$(basename ${dset}_trials_f_all)/trials ./data/$(basename ${dset}_trials_m_all)/trials > ./data/$(basename ${dset}_trials_all)/trials

  cp ./data/$(basename ${dset}_trials_f_all)/utt2spk ./data/$(basename ${dset}_trials_f_all_anon)
  cp ./data/$(basename ${dset}_trials_f_all)/trials ./data/$(basename ${dset}_trials_f_all_anon)

  cut -d' ' -f2 ./data/$(basename ${dset}_trials_f)/trials | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ./data/$(basename ${dset}_trials_f_all_anon) ./data/$(basename ${dset}_trials_f_anon) || exit 1
  cp ./data/$(basename ${dset}_trials_f)/trials ./data/$(basename ${dset}_trials_f_anon) || exit 1

  cut -d' ' -f2 ./data/$(basename ${dset}_trials_f_common)/trials | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ./data/$(basename ${dset}_trials_f_all_anon) ./data/$(basename ${dset}_trials_f_common_anon) || exit 1
  cp ./data/$(basename ${dset}_trials_f_common)/trials ./data/$(basename ${dset}_trials_f_common_anon) || exit 1

  cp ./data/$(basename ${dset}_trials_m_all)/utt2spk ./data/$(basename ${dset}_trials_m_all_anon)
  cp ./data/$(basename ${dset}_trials_m_all)/trials ./data/$(basename ${dset}_trials_m_all_anon)

  cut -d' ' -f2 ./data/$(basename ${dset}_trials_m)/trials | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ./data/$(basename ${dset}_trials_m_all_anon) ./data/$(basename ${dset}_trials_m_anon) || exit 1
  cp ./data/$(basename ${dset}_trials_m)/trials ./data/$(basename ${dset}_trials_m_anon) || exit 1

  cut -d' ' -f2 ./data/$(basename ${dset}_trials_m_common)/trials | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ./data/$(basename ${dset}_trials_m_all_anon) ./data/$(basename ${dset}_trials_m_common_anon) || exit 1
  cp ./data/$(basename ${dset}_trials_m_common)/trials ./data/$(basename ${dset}_trials_m_common_anon) || exit 1

  cp ./data/$(basename ${dset}_enrolls)/utt2spk ./data/$(basename ${dset}_enrolls_anon)
  cp ./data/$(basename ${dset}_enrolls)/enrolls ./data/$(basename ${dset}_enrolls_anon)

done
rm $temp

grep "data/libri_test/wav/" -r ./data --files-with-matches | \
        xargs sed -i "s|data/libri_test/wav/|wav_clear/libri_test/wav/|g"

grep "data/libri_dev/wav/" -r ./data --files-with-matches | \
        xargs sed -i "s|data/libri_dev/wav/|wav_clear/libri_dev/wav/|g"

grep "data/vctk_dev/wav/" -r ./data --files-with-matches | \
        xargs sed -i "s|data/vctk_dev/wav/|wav_clear/vctk_dev/wav/|g"

grep "data/vctk_test/wav/" -r ./data --files-with-matches | \
        xargs sed -i "s|data/vctk_test/wav/|wav_clear/vctk_test/wav/|g"
