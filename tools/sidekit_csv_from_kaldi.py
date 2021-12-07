# -*- coding: utf8 -*-
import os
import csv
import torchaudio
from tqdm import tqdm
from argparse import ArgumentParser

def main():
    parser = ArgumentParser("Convert kaldi directory to sidekit csv")
    parser.add_argument("--kaldi-data-path", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--database-name", type=str, default="")
    args = parser.parse_args()
    kaldi_data_path = args.kaldi_data_path
    out_csv_path = args.out_csv
    database_name = args.database_name if args.database_name != "" else os.path.basename(kaldi_data_path)

    # Read spk2gender file
    spk2gender_dict = {}
    with open(os.path.join(kaldi_data_path, "spk2gender"), "r") as spk2gender_file:
        for line in spk2gender_file:
            split_line = line.split(" ")
            spk2gender_dict[split_line[0]] = split_line[1].replace("\n", "")

    # Read reco2dur file if exists
    reco2dur_dict = {}
    reco2dir_path = os.path.join(kaldi_data_path, "reco2dur")
    if os.path.exists(reco2dir_path):
        with open(reco2dir_path, "r") as reco2dur_file:
            for line in reco2dur_file:
                split_line = line.split(" ")
                spk = split_line[0].split("-")[0]  # Remove sub-id of speaker if exists (useful for librispeech dataset)
                reco2dur_dict[spk] = split_line[1].replace("\n", "")

    spk_list = []
    out_csv_file = open(out_csv_path, "w", newline="")
    csv_writer = csv.writer(
        out_csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )

    # Write header
    csv_writer.writerow(
        [
            "speaker_idx",
            "database",
            "speaker_id",
            "start",
            "duration",
            "file_id",
            "gender",
        ]
    )

    # Count lines in scp file
    scp_path = os.path.join(kaldi_data_path, "wav.scp")
    num_lines = sum(1 for line in open(scp_path, "r"))

    scp_file = open(scp_path, "r")
    iter_scp = tqdm(scp_file, total=num_lines)
    for line in iter_scp:
        split_line = line.split(" ")
        utt_id = split_line[0]
        if len(split_line) == 2:
            # Scp file without command inside
            file_path = split_line[1]
        else:
            # Scp file with command inside.
            # Read all split to find text with '/' indicating it's a path
            file_path = ""
            for word in split_line:
                if '/' in word:
                    file_path = word
            if file_path == "":
                raise FileNotFoundError("No filepath found in line : ", line)
        spk_id = utt_id.split("-")[0]
        if spk_id not in spk_list:
            spk_list.append(spk_id)
        spk_idx = spk_list.index(spk_id)

        start = 0
        if len(reco2dur_dict) > 0:
            # Load duration from existing reco2dur file
            duration = reco2dur_dict[utt_id]
        else:
            # No reco2dur file, loading duration with torchaudio
            duration = calculate_duration(file_path)

        file_id = os.path.splitext(file_path)[0]
        gender = spk2gender_dict[spk_id]


        csv_writer.writerow(
            [
                spk_idx,
                database_name,
                spk_id,
                start,
                duration,
                file_id,
                gender,
            ]
        )

    scp_file.close()
    out_csv_file.close()

def calculate_duration(file_path):
    try:
        audio_info = torchaudio.info(file_path)
        duration = audio_info.num_frames / audio_info.sample_rate
    except Exception:
        print("Failed to load info of:", file_path)
        duration = 0

    return duration

if __name__ == "__main__":
    main()