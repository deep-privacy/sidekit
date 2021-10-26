#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# The script downloads the librispeech_360 dataset
# Requirement: ffmpeg and wget running on a Linux system.

# Inspired from:  https://raw.githubusercontent.com/clovaai/voxceleb_trainer/master/dataprep.py

import argparse
import os
import sys
import subprocess
import pdb
import hashlib
import time
import glob
import tarfile
from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile
from pathlib import Path

import torchaudio
import csv

## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description="Librispeech dataset preparation")

parser.add_argument("--save-path", type=str, default="data", help="Target directory")

parser.add_argument(
    "--download", dest="download", action="store_true", help="Enable download"
)
parser.add_argument(
    "--download-augment",
    dest="augment",
    action="store_true",
    help="Download and extract augmentation files",
)
parser.add_argument(
    "--make-train-csv",
    dest="make_train_csv",
    action="store_true",
    help="Create the training sidekit csv",
)
parser.add_argument(
    "--make-csv-augment-reverb",
    dest="make_aug_csv_reverb",
    action="store_true",
    help="Create the dataset aug sidekit csv",
)
parser.add_argument(
    "--make-csv-augment-noise",
    dest="make_aug_csv_noise",
    action="store_true",
    help="Create the dataset aug sidekit csv",
)

## args for --make-*-csv
parser.add_argument(
    "--from",
    default="./data/LibriSpeech",
    dest="_from",
    type=str,
    help="Path to the root of the dataset",
)
parser.add_argument(
    "--out-csv", default="list/libri.csv", type=str, help="File to the output csv"
)
parser.add_argument(
    "--fullpath",
    type=str,
    default="True",
    help='List training audio files with their full path, otherwise relative to "root"',
)
parser.add_argument(
    "--filter-dir",
    dest="filter_dataset",
    type=str,
    default="train-clean-360",
    help="List of dir of process. Must be the exact match of one directory (no '/' allow). Default: 'train-clean-360' delimited with ','",
)


## ========== ===========
## MD5SUM
## ========== ===========
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


## ========== ===========
## Download with wget
## ========== ===========
def download(args, lines):
    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split("/")[-1]

        if os.path.exists("%s/%s" % (args.save_path, outfile)):
            print("File '%s' exist checking md5.." % outfile)
            md5ck = md5("%s/%s" % (args.save_path, outfile))
            if md5ck == md5gt:
                print("Skipping already downloaded '%s' md5 match." % outfile)
                continue

        ## Download files
        out = subprocess.call(
            "wget --no-check-certificate %s -O %s/%s" % (url, args.save_path, outfile),
            shell=True,
        )
        if out != 0:
            print(
                "Download failed %s please retry!\nIf download fails repeatedly, use alternate URL."
                % outfile
            )
            sys.exit(1)

        ## Check MD5
        md5ck = md5("%s/%s" % (args.save_path, outfile))
        if md5ck == md5gt:
            print("Checksum successful %s." % outfile)
        else:
            raise Warning("Checksum failed %s." % outfile)


## ========== ===========
## Extract zip files
## ========== ===========
def full_extract(args, fname):
    print("Extracting %s" % fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(args.save_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, "r") as zf:
            zf.extractall(args.save_path)


## ========== ===========
## Partially extract zip files
## ========== ===========
def part_extract(args, fname, target):
    print("Extracting %s" % fname)
    with ZipFile(fname, "r") as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, args.save_path)
            # pdb.set_trace()
            # zf.extractall(args.save_path)


## ========== ===========
## Split MUSAN for faster random access
## ========== ===========
def split_musan(args):
    files = glob.glob("%s/musan/*/*/*.wav" % args.save_path)

    audlen = 16000 * 5
    audstr = audlen

    for idx, file in enumerate(files):
        fs, aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace("/musan/", "/musan_split/"))[0]
        os.makedirs(writedir)
        for st in range(0, len(aud) - audlen, audstr):
            wavfile.write(writedir + "/%05d.wav" % (st / fs), fs, aud[st : st + audlen])

        print(idx, file)


## ========== ===========
## Create sidekit csv file
##  FOR AUGMENTATAION
## ========== ===========
def make_aug_csv_reverb(root_data, out_filepath, fullpath):
    with open(out_filepath, "w", newline="") as out_csv_file:
        csv_writer = csv.writer(
            out_csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )

        # Write header
        csv_writer.writerow(
            [
                "channel",
                "database",
                "file_id",
                "type",
            ]
        )
        wav_count = 0
        pbar = tqdm(os.walk(root_data))
        for root, dirs, files in pbar:
            if Path(root).parent == Path(root_data):
                dataset = root.split("/")[-1]
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1] == ".wav":
                    wav_count += 1
                    pbar.set_description(f"wav count : {wav_count}")
                    try:
                        audio_info = torchaudio.info(file_path)
                    except Exception:
                        print("failed to load info of:", file_path)
                        continue
                    if fullpath.lower() == "true":
                        file_id = os.path.realpath(file_path)
                    else:
                        # Remove file root_data
                        file_id = file_path.replace(root_data, "")
                        # Remove first slash if present (it is not root_data)
                        file_id = file_id[1:] if file_id[0] == "/" else file_id

                    csv_writer.writerow([1.0, "REVERB", file_id, dataset])


def make_aug_csv_noise(root_data, out_filepath, fullpath):
    with open(out_filepath, "w", newline="") as out_csv_file:
        csv_writer = csv.writer(
            out_csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )

        # Write header
        csv_writer.writerow(
            [
                "database",
                "type",
                "file_id",
                "start",
                "duration",
            ]
        )
        wav_count = 0
        pbar = tqdm(os.walk(root_data))
        for root, dirs, files in pbar:
            if Path(root).parent == Path(root_data):
                dataset = root.split("/")[-1]
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1] == ".wav":
                    wav_count += 1
                    pbar.set_description(f"wav count : {wav_count}")
                    try:
                        audio_info = torchaudio.info(file_path)
                        duration = audio_info.num_frames / audio_info.sample_rate
                    except Exception:
                        print("failed to load info of:", file_path)
                        continue
                    if fullpath.lower() == "true":
                        # Remove only file extension
                        file_id = os.path.splitext(os.path.realpath(file_path))[0]
                    else:
                        # Remove file extension and file root_data
                        file_id = os.path.splitext(file_path)[0].replace(root_data, "")
                        # Remove first slash if present (it is not root_data)
                        file_id = file_id[1:] if file_id[0] == "/" else file_id

                    csv_writer.writerow(
                        [
                            "musan",
                            dataset,
                            file_id,
                            0.0,
                            duration,
                        ]
                    )


## ========== ===========
## Create sidekit csv file
##    FOR LIBRISPEECH
## ========== ===========
def make_train_csv(root_data, out_filepath, fullpath, filter_dataset):
    # Retrieve gender for Librispeech speakers
    spk_file = open(os.path.join(root_data, "SPEAKERS.TXT"), "r")
    spk_gender_dict = {}
    for line in spk_file:
        if line[0] != ";":
            split_line = line.split("|")
            spk_gender_dict[split_line[0].strip()] = split_line[1].strip().lower()

    # Browse directories to retrieve list of audio files
    spk_list = []
    with open(out_filepath, "w", newline="") as out_csv_file:
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

        pbar = tqdm(os.walk(root_data))
        for root, dirs, files in pbar:
            dataset = root.split("/")
            _continue = True
            for _filter in filter_dataset:
                if _filter in dataset:
                    _continue = False
                    break

            pbar.set_description(f"spk count: {len(spk_list)} scaning: {root}")
            if _continue:
                continue

            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1] == ".flac":
                    spk_id = file.split("-")[0]
                    if spk_id not in spk_list:
                        spk_list.append(spk_id)
                    spk_idx = spk_list.index(spk_id)
                    start = 0
                    audio_info = torchaudio.info(file_path)
                    duration = audio_info.num_frames / audio_info.sample_rate
                    if fullpath.lower() == "true":
                        # Remove only file extension
                        file_id = os.path.splitext(os.path.realpath(file_path))[0]
                    else:
                        # Remove file extension and file root_data
                        file_id = os.path.splitext(file_path)[0].replace(root_data, "")
                        # Remove first slash if present (it is not root_data)
                        file_id = file_id[1:] if file_id[0] == "/" else file_id
                    gender = spk_gender_dict[spk_id]

                    csv_writer.writerow(
                        [spk_idx, dataset[-3], spk_id, start, duration, file_id, gender]
                    )


## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":
    args = parser.parse_args()

    if args.make_aug_csv_reverb:
        if not os.path.exists(args._from):
            raise ValueError(f"Dataset directory '{args._from}' does not exist.")

        if args._from == parser.get_default("_from"):
            # change default to downloaded reverb
            args._from = "./data/RIRS_NOISES"

        if args.out_csv == parser.get_default("out_csv"):
            # change default to output reverb list
            args.out_csv = "list/reverb.csv"

        make_aug_csv_reverb(args._from, args.out_csv, args.fullpath)
        sys.exit(0)

    if args.make_aug_csv_noise:
        if not os.path.exists(args._from):
            raise ValueError(f"Dataset directory '{args._from}' does not exist.")

        if args._from == parser.get_default("_from"):
            # change default to downloaded reverb
            args._from = "./data/musan_split"

        if args.out_csv == parser.get_default("out_csv"):
            # change default to output reverb list
            args.out_csv = "list/musan.csv"

        make_aug_csv_noise(args._from, args.out_csv, args.fullpath)
        sys.exit(0)

    if args.make_train_csv:
        if not os.path.exists(args._from):
            raise ValueError(f"Dataset directory '{args._from}' does not exist.")

        make_train_csv(
            args._from, args.out_csv, args.fullpath, args.filter_dataset.split(",")
        )
        sys.exit(0)

    if not os.path.exists(args.save_path):
        raise ValueError(f"Target directory '{args.save_path}' does not exist.")

    f = open("list/files.txt", "r")
    files = f.readlines()
    f.close()

    f = open("list/augment.txt", "r")
    augfiles = f.readlines()
    f.close()

    if args.augment:
        download(args, augfiles)
        #  part_extract(args,os.path.join(args.save_path,'rirs_noises.zip'),['RIRS_NOISES/simulated_rirs/mediumroom','RIRS_NOISES/simulated_rirs/smallroom'])
        part_extract(
            args, os.path.join(args.save_path, "rirs_noises.zip"), ["RIRS_NOISES"]
        )
        full_extract(args, os.path.join(args.save_path, "musan.tar.gz"))
        split_musan(args)

    if args.download:
        download(args, files)
        for line in files:
            outfile = line.split()[0].split("/")[-1]
            full_extract(args, os.path.join(args.save_path, outfile))
