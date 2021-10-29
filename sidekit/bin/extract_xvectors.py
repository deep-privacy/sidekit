import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torchaudio
from sidekit.nnet.xvector import Xtractor

import os
import io
import argparse
import subprocess
import json

import kaldiio
import soundfile

def read_wav_scp(wav_scp):
    """Reads wav.scp file and returns a dictionary

    Args:
        wav_scp: a string, contains the path to wav.scp

    Returns:
        utt2wav: a dictionary, keys are the first column of wav.scp
            and values are the second column
    """
    utt2wav = {}
    with open(wav_scp) as ipf:
        for line in ipf:
            lns = line.strip().split()
            uttname = lns[0]
            utt2wav[uttname] = lns[1:]
    return utt2wav

def prepare(wav):
    """Reads a wav.scp entry like kaldi with embeded unix command
    and returns a pytorch tensor like it was open with torchaudio.load()
    (within some tolerance due to numerical precision)

    signal, _ = torchaudio.load("XX/1272-128104-0000.flac")
    signalv2 = prepare(['flac', '-c', '-d', '-s', 'XX/1272-128104-0000.flac', "|"])
    signalv3 = prepare(['XX/1272-128104-0000.flac'])

    print("all close:", torch.allclose(signal, signalv2, rtol=1e-1))
    print("all close:", torch.allclose(signal, signalv3, rtol=1e-1))

    Args:
        wav: a list containing the scp entry

    Returns:
        feats_torch torch.tensor dtype float32
    """
    wav = ' '.join(wav)
    if wav.strip().endswith("|"):
        devnull = open(os.devnull, 'w')
        try:
            wav_read_process = subprocess.Popen(
                wav.strip()[:-1],
                stdout=subprocess.PIPE,
                shell=True,
                stderr=devnull
            )
            sample, sr = soundfile.read(
                io.BytesIO(wav_read_process.communicate()[0]),
            )
        except Exception as e:
            raise IOError("Error processing wav file: {}\n{}".format(wav, e))
    else:
        sample, sr = soundfile.read(wav)
    feats_torch = torch.tensor(sample, dtype=torch.float32, requires_grad=False)
    return feats_torch, sr

def load_model(model_path, device):
    device = torch.device(device)
    model_config = torch.load(model_path, map_location=device)

    model_opts = model_config["model_archi"]
    if "embedding_size" not in model_opts:
        model_opts["embedding_size"] = 256
    xtractor = Xtractor(model_config["speaker_number"],
                     model_archi=model_opts["model_type"],
                     loss=model_opts["loss"]["type"],
                     embedding_size=model_opts["embedding_size"])

    xtractor.load_state_dict(model_config["model_state_dict"], strict=True)
    xtractor = xtractor.to(device)
    xtractor.eval()
    return xtractor, model_config

@torch.no_grad()
def main(xtractor, kaldi_wav_scp, out_file, device, vad, num_samples_per_window, min_silence_samples, model_sample_rate):
    device = torch.device(device)

    utt2wav = read_wav_scp(kaldi_wav_scp)
    out_ark = os.path.realpath(os.path.join(os.path.dirname(out_file), os.path.splitext(os.path.basename(out_file))[0]))

    if vad:
        torch.set_num_threads(1) # faster vad on cpu
        torch.backends.quantized.engine = 'qnnpack' # compatibility

        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad:a345715',
                                      model='silero_vad_mini')

        (get_speech_ts,
         get_speech_ts_adaptive,
         _,
         read_audio,
         _,
         _,
         collect_chunks) = utils

        vad_cache = os.path.splitext(out_file)[0] + "_vad.json"
        cache_speech_timestamps = {}
        if os.path.isfile(vad_cache):
            with open(vad_cache, 'r') as vad_file:
                cache_speech_timestamps = json.load(vad_file)


    with kaldiio.WriteHelper(f'ark,scp:{out_ark}.ark,{os.path.realpath(out_file)}') as writer:
        for key, wav in utt2wav.items():
            signal, sr = prepare(wav)
            if vad:
                signal_for_vad = signal
                if sr != model_sample_rate:
                    signal_for_vad = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=1600)(signal)
                if key in cache_speech_timestamps:
                    speech_timestamps = cache_speech_timestamps[key]
                else:
                    speech_timestamps = get_speech_ts_adaptive(signal_for_vad.mean(dim=0, keepdim=True),
                                                               model,
                                                               step=500,
                                                               num_samples_per_window=num_samples_per_window,
                                                               min_silence_samples=min_silence_samples)
                    if len(speech_timestamps) == 0:
                        speech_timestamps = [{"start":0, "end":len(signal)}]
                signal = collect_chunks(speech_timestamps, signal_for_vad)
                cache_speech_timestamps[key] = speech_timestamps

            signal = signal.to(device)
            if sr != model_sample_rate:
                signal = torchaudio.transforms.Resample(orig_freq=sr,
                                               new_freq=model_sample_rate)(signal)
            _, vec = xtractor(signal, is_eval=True)
            writer(key, vec.detach().cpu().numpy())

        if not os.path.isfile(vad_cache):
            with open(vad_cache, "w") as vad_file:
                json.dump(cache_speech_timestamps, vad_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the x-vectors given a sidekit model")
    parser.add_argument("--model", type=str, help="SideKit model", required=True)
    parser.add_argument("--sample-rate", type=int, help="Must match SideKit SR model", default=16000)
    parser.add_argument("--vad", action='store_true', help="Apply vad before extracting the x-vector")
    parser.add_argument("--vad-num-samples-per-window", type=int, default=2000, help="Number of samples in each window, (2000 -> 125ms) per window. Check https://github.com/snakers4/silero-vad for more info")
    parser.add_argument("--vad-min-silence-samples", type=int, default=1500, help="Minimum silence duration in samples between to separate speech chunks, (1500). Check https://github.com/snakers4/silero-vad for more info")
    parser.add_argument("--wav-scp", type=str, required=True)
    parser.add_argument("--out-scp", type=str, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="The device (cpu or cuda:0) to run the inference")
    args = parser.parse_args()

    assert os.path.isfile(args.model), "NO SUCH FILE: %s" % args.model
    assert os.path.isfile(args.wav_scp), "NO SUCH FILE: %s" % args.wav_scp
    assert os.path.isdir(os.path.dirname(args.out_scp)), "NO SUCH DIRECTORY: %s" % args.out_scp
    # If cuda device is requested, check if cuda is available
    args.device = args.device.strip().lower()
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available, check configuration or run on cpu (--device cpu)"
    xtractor, model_config = load_model(args.model, args.device)
    main(xtractor, args.wav_scp, args.out_scp, args.device, args.vad, args.vad_num_samples_per_window, args.vad_min_silence_samples, args.sample_rate)
