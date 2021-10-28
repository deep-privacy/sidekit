import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
import os
import subprocess
import pandas
import numpy
import torch
import torchaudio
import torch.multiprocessing as mp
from tqdm import tqdm
from argparse import ArgumentParser

torch.set_num_threads(1) # faster vad on cpu
torch.backends.quantized.engine = 'qnnpack' # compatibility

def main(df, out_csv, out_audio_dir, audio_dir, extension_name, nj, num_samples_per_window, min_silence_samples):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad:a345715',
                                  model='silero_vad_mini')

    (get_speech_ts,
     get_speech_ts_adaptive,
     save_audio,
     read_audio,
     state_generator,
     single_audio_stream,
     collect_chunks) = utils

    def job(iter, out_csv, log_one=False):

        out_cvs_list = []
        for i, row in iter:
            try:
                wav = read_audio(audio_dir + "/" + row["file_id"] + "." + extension_name)
                # Experimental Adaptive method, algorithm selects thresholds itself
                speech_timestamps = get_speech_ts_adaptive(wav, model,
                                                           step=500,
                                                           num_samples_per_window=num_samples_per_window,
                                                           min_silence_samples=min_silence_samples)

                if log_one:
                    print("Sample of one VAD application:", row["file_id"] + "." + extension_name, speech_timestamps)
                    log_one = False

                row["file_id"] = os.path.realpath(f"{out_audio_dir}/vad_{os.path.basename(row['file_id'])}")
                #  # merge all speech chunks to one audio
                vad_wav = collect_chunks(speech_timestamps, wav)
                row["duration"] = len(vad_wav) / 16000
                save_audio(row["file_id"]+"."+extension_name, vad_wav, 16000)

            except Exception:
                print(f"failed to process: {os.path.basename(row['file_id'])}, not applying vad for this file")

            out_cvs_list.append(row)

        pandas.DataFrame(out_cvs_list).to_csv(out_csv, index=False)

    df = numpy.array_split(df, nj)
    processes = []
    for i in range(nj):

        iter = df[i].iterrows()
        if i == 0:
            iter = tqdm(iter, total=len(df[i]))

        p = mp.Process(target=job, args=(iter, out_csv+f"_job{i}", i == 0))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    subprocess.call(
            f"awk '!a[$0]++' {out_csv}_job* > {out_csv}",
            shell=True,
        )
    subprocess.call(
            f"\\rm {out_csv}_job*",
            shell=True,
        )


if __name__ == "__main__":
    parser = ArgumentParser("Apply Silero-vad on a sidekit training csv file")
    parser.add_argument("--in-csv", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--out-audio-dir", type=str, required=True)
    parser.add_argument("--audio-dir", type=str, default="/")
    parser.add_argument("--extension-name", type=str, default="wav")
    parser.add_argument("--nj", type=int, default=8)
    parser.add_argument("--num-samples-per-window", type=int, default=2000,
                        help="Number of samples in each window, (2000 -> 125ms) per window. Check https://github.com/snakers4/silero-vad for more info")
    parser.add_argument("--min-silence-samples", type=int, default=2000,
                        help="Minimum silence duration in samples between to separate speech chunks, (2000). Check https://github.com/snakers4/silero-vad for more info")
    args = parser.parse_args()

    assert os.path.isfile(args.in_csv), "NO SUCH FILE: %s" % args.in_csv
    if not os.path.isdir(args.out_audio_dir):
        print("Creating new directory: {args.out_audio_dir}")
        os.mkdir(args.out_audio_dir)

    print("Applying vad for:", args.in_csv)
    df = pandas.read_csv(args.in_csv)
    main(df, args.out_csv, args.out_audio_dir, args.audio_dir, args.extension_name, args.nj, args.num_samples_per_window, args.min_silence_samples)
