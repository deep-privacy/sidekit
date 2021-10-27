import torch
import sys
import pandas
from argparse import ArgumentParser

torch.backends.quantized.engine = 'qnnpack'

def main(in_csv, out_csv):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad:a345715',
                                  model='silero_vad')

    (get_speech_ts,
     get_speech_ts_adaptive,
     save_audio,
     read_audio,
     state_generator,
     single_audio_stream,
     collect_chunks) = utils

    df = pandas.read_csv(in_csv)
    print(df["file_id"][0])

    for id in range(len(df)):
        wav = read_audio(df["file_id"][id] + ".flac")
        speech_timestamps = get_speech_ts(wav, model,
                                          num_steps=4)
        print(speech_timestamps)

        # or use: Experimental Adaptive method, algorithm selects thresholds itself
        speech_timestamps = get_speech_ts_adaptive(wav, model, step=500, num_samples_per_window=4000)
        print(speech_timestamps)

        # merge all speech chunks to one audio
        #  save_audio('only_speech.wav',
                   #  collect_chunks(speech_timestamps, wav), 16000)

if __name__ == "__main__":
    parser = ArgumentParser("Apply Silero-vad on a sidekit training csv file")
    parser.add_argument("--in-csv", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    main(args.in_csv, args.out_csv)
