import torch
import torchaudio
from sidekit.nnet.xvector import Xtractor
import argparse


@torch.no_grad()
def main(model_path, device):
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
    #  signal, sr = torchaudio.load("/home/hnourtel/Documents/Test/wav/Ses01M_Ses01M_impro04_M010_anon.wav")
    signal = torch.rand(1, 3000)
    signal = signal.to(device)
    _, vec = xtractor(signal, is_eval=True)
    print(vec.size())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the x-vectors given a sidekit model")
    parser.add_argument("--model", type=str, help="SideKit model", required=True)
    parser.add_argument("--device", default="cuda"if torch.cuda.is_available() else "cpu", type=str, help="The device (cpu or cuda:0) to run the inferance")
    args = parser.parse_args()
    main(args.model, args.device)
