# -*- coding: utf8 -*-
import os
import numpy as np
from sidekit.bosaris import Ndx, IdMap, Key

def main():
    enrolls_dir = "/home/hnourtel/talcStorage/sidekit/egs/vpc2020_decode/data/libri_test_enrolls"
    trials_dir = "/home/hnourtel/talcStorage/sidekit/egs/vpc2020_decode/data/libri_test_trials_all"
    # enrolls_dir = "libri_test_enrolls"
    # trials_dir = "libri_test_trials_all"

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
    trials_dict = {}
    with open(os.path.join(trials_dir, "trials")) as trials_file:
        for line in trials_file:
            split_line = line.rstrip("\n").split(" ")
            trials_dict.setdefault(split_line[0], []).append((split_line[1], split_line[2])) # key = spk, val = [(utt, target type)]

    modelset = []
    segset = []
    non = []
    tar = []
    with open(os.path.join(trials_dir, "utt2spk")) as utt2spk_trials_file:
        for line in utt2spk_trials_file:
            split_line = line.split(" ")
            utt = split_line[0]
            segset.append(utt)

    for model in left_ids:
        modelset.append(model)
        spk_model = model.split("-")[0]
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

    key = Key()
    key.modelset = np.array(modelset)
    key.segset = np.array(segset)
    key.tar = np.array(tar)
    key.non = np.array(non)

    # Ndx
    ndxData = key.to_ndx()

    # Write test file in text format
    idmap.write_txt("libri_test_idmap.txt")
    key.write_txt("libri_test_key.txt")
    ndxData.save_txt("libri_test_ndx.txt")

    # Write test file in h5 format
    idmap.write("libri_test_idmap.h5")
    key.write("libri_test_key.h5")
    ndxData.write("libri_test_ndx.h5")

    print("End")

if __name__ == "__main__":
    main()
