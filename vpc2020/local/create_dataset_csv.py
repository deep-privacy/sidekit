# -*- coding: utf8 -*-
import csv
import os
import torchaudio
from argparse import ArgumentParser

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--libri_root', type=str, help='Path to the root of Librispeech dataset')
    arg_parser.add_argument('--out_filepath', type=str, help='Path to the output csv file')
    arg_parser.add_argument('--add_root_to_file_id', type=bool, default=True, help='Add libri_root to the file_id field in output csv file')
    arg_parser.add_argument('--filter_subdataset', type=list, default=[], help='List of subdataset of Librispeech to process. Leave empty to process all subdataset')

    args = arg_parser.parse_args()
    libri_root = args.libri_root
    out_filpath = args.out_filepath
    add_root_to_file_id = args.add_root_to_file_id
    filter_subdataset = args.filter_subdataset

    # Retrieve gender for Librispeech speakers
    spk_file = open(os.path.join(libri_root, "SPEAKERS.TXT"), "r")
    spk_gender_dict = {}
    for line in spk_file:
        if line[0] != ";":
            split_line = line.split("|")
            spk_gender_dict[split_line[0].strip()] = split_line[1].strip().lower()

    # Browse directories to retrieve list of audio files
    spk_list = []
    with open(out_filpath, 'w', newline='') as out_csv_file:
        csv_writer = csv.writer(out_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Write header
        csv_writer.writerow(["speaker_idx", "database", "speaker_id", "start", "duration", "file_id", "gender"])

        for root, dirs, files in os.walk(libri_root):
            if root == libri_root:
                # Exclude subdataset directories if not in filter_subdataset list.
                # Only done once on libri_root directory list
                if len(filter_subdataset) > 0:
                    [dirs.remove(d) for d in list(dirs) if d not in filter_subdataset]
            for file in files:
                if file.split(".")[-1] == "flac":
                    spk_id = file.split("-")[0]
                    if spk_id not in spk_list:
                        spk_list.append(spk_id)
                        print("spk count : ", len(spk_list))
                    spk_idx = spk_list.index(spk_id)
                    dataset = root.split("/")[-3]
                    start = 0
                    file_path = os.path.join(root, file)
                    audio_info = torchaudio.info(file_path)
                    duration = audio_info.num_frames / audio_info.sample_rate
                    if add_root_to_file_id:
                        file_id = file_path.split(".")[0]  # Remove only file extension
                    else:
                        file_id = file_path.split(".")[0].replace(libri_root, "")  # Remove file extension and file root
                        file_id = file_id[1:] if file_id[0] == "/" else file_id  # Remove first slash if present (it is not root)
                    gender = spk_gender_dict[spk_id]

                    csv_writer.writerow([spk_idx, dataset, spk_id, start, duration, file_id, gender])

if __name__ == "__main__":
    main()
