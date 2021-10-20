# -*- coding: utf8 -*-
import csv
import os
import torchaudio

def main():
    out_filname = "list/libri360.csv"
    libri_root = "/home/hnourtel/talcStorage/Librispeech_train_clean_360/LibriSpeech"
    filter_dir = ["train-clean-360"]
    #TODO : ajouter filtre dossier pour ne pas tout parcourir

    # Retrieve gender for Librispeech speakers
    spk_file = open(os.path.join(libri_root, "SPEAKERS.TXT"), "r")
    spk_gender_dict = {}
    for line in spk_file:
        if line[0] != ";":
            split_line = line.split("|")
            spk_gender_dict[split_line[0].strip()] = split_line[1].strip().lower()

    spk_list = []
    # Browse directories to retrieve list audio files
    with open(out_filname, 'w', newline='') as out_csv_file:
        csv_writer = csv.writer(out_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Write header
        csv_writer.writerow(["speaker_idx", "database", "speaker_id", "start", "duration", "file_id", "gender"])

        for root, dirs, files in os.walk(libri_root):
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
                    file_id = file_path.split(".")[0] # Remove file extension
                    gender = spk_gender_dict[spk_id]

                    csv_writer.writerow([spk_idx, dataset, spk_id, start, duration, file_id, gender])

if __name__ == "__main__":
    main()