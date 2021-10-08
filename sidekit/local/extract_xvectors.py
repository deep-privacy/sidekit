import torch
import torchaudio
from sidekit.nnet.xvector import Xtractor
import argparse


@torch.no_grad()
def main(model_path, device):
    device = torch.device(device)
    model = torch.load(model_path, map_location=device)

    model_opts = model["model_archi"]
    model = Xtractor(model["speaker_number"],
                     model_archi=model_opts["model_type"],
                     loss=model_opts["loss"]["type"],
                     embedding_size=model_opts["embedding_size"])

    model.load_state_dict(model["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    #  signal, sr = torchaudio.load("/home/hnourtel/Documents/Test/wav/Ses01M_Ses01M_impro04_M010_anon.wav")
    signal = torch.rand(1, 3000)
    _, vec = model(signal, is_eval=True)
    print(vec.size())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract the x-vectors given a sidekit model")
    parser.add_argument("model", type=str)
    parser.add_argument("--device", default="cuda"if torch.cuda.is_available() else "cpu", type=str, help="The device (cpu or cuda:0) to run the inferance")

    args = parser.parse_args()
    main(args["model"])
