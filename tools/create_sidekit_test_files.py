# -*- coding: utf8 -*-
import os
import numpy as np
from sidekit.bosaris import IdMap, Key
import argparse

def main():

    # Retrieve arguments
    parser = argparse.ArgumentParser(description="Creating tests files for Sidekit training from Kaldi format dataset")
    parser.add_argument('--enrolls-dir', type=str, help='Path to the enrollment directory (Kaldi format)')
    parser.add_argument('--trials-dir', type=str, help='Path to the trials directory (Kaldi format)')
    parser.add_argument('--out-dir', type=str, default='.', help='Path to the output directory')
    parser.add_argument('--out-file-prefix', type=str, help='Prefix for all output files')
    parser.add_argument('--out-format', type=str, default="h5", choices=["h5", "txt"], help='Format of output files')

    args = parser.parse_args()

    enrolls_dir = args.enrolls_dir
    trials_dir = args.trials_dir
    out_dir = args.out_dir
    out_prefix = args.out_file_prefix
    out_format = args.out_format

    # IDmap
    left_ids = []
    right_ids = []
    with open(os.path.join(enrolls_dir, "enrolls")) as enrolls_file:
        for line in enrolls_file:
            cleared_line = line.rstrip("\n")
            left_ids.append(cleared_line)
            right_ids.append(cleared_line)

    idmap = IdMap()
    idmap.set(np.array(left_ids), np.array(right_ids))

    # Key
    # Read utt2spk file from enrolls directory to have the matching model/speaker (model's name is the name of the utterance)
    enrolls_utt2spk = {}
    with open(os.path.join(enrolls_dir, "utt2spk")) as enrolls_utt2spk_file:
        for line in enrolls_utt2spk_file:
            split_line = line.rstrip("\n").split(" ")
            enrolls_utt2spk[split_line[0]] = split_line[1]

    # Read trials file and store results in dict with all utterance for a given speaker
    trials_dict = {}
    with open(os.path.join(trials_dir, "trials")) as trials_file:
        for line in trials_file:
            split_line = line.rstrip("\n").split(" ")
            trials_dict.setdefault(split_line[0], []).append((split_line[1], split_line[2])) # key = spk, val = [(utt, target type)]

    # Read all segments used for trial
    segset = []
    with open(os.path.join(trials_dir, "utt2spk")) as utt2spk_trials_file:
        for line in utt2spk_trials_file:
            split_line = line.split(" ")
            utt = split_line[0]
            segset.append(utt)

    # For each model in Idmap, retrieve trial utterances for the model's speaker and fill target/non-target lists
    modelset = []
    non = []
    tar = []
    for model in left_ids:
        modelset.append(model)
        spk_model = enrolls_utt2spk[model]
        new_non = [False] * len(segset)
        new_tar = [False] * len(segset)
        for trial_utt, trial_tar in trials_dict[spk_model]:
            seg_idx = segset.index(trial_utt)
            if trial_tar == "target":
                new_tar[seg_idx] = True
            else:
                new_non[seg_idx] = True
        non.append(np.array(new_non))
        tar.append(np.array(new_tar))

    key = Key.create(np.array(modelset), np.array(segset), np.array(tar), np.array(non))

    # Ndx from Idmap
    ndxData = key.to_ndx()

    # Writing output files
    if out_format == "txt":
        idmapWriteFunc = idmap.write_txt
        keyWriteFunc = key.write_txt
        ndxWriteFunc = ndxData.save_txt
    else:
        idmapWriteFunc = idmap.write
        keyWriteFunc = key.write
        ndxWriteFunc = ndxData.write
    idmapWriteFunc(os.path.join(out_dir, out_prefix + "_idmap." + out_format))
    keyWriteFunc(os.path.join(out_dir, out_prefix + "_key." + out_format))
    ndxWriteFunc(os.path.join(out_dir, out_prefix + "_ndx." + out_format))

    print("Sidekit test files successfully created")

if __name__ == "__main__":
    main()
