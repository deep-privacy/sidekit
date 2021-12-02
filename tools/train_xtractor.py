# -*- coding: utf8 -*-
import sidekit
from argparse import ArgumentParser


def main():
    parser = ArgumentParser("SideKit xvector training conf")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        metavar="N",
        help="PyTorch DDP Local process rank.",
    )
    parser.add_argument("--dataset", type=str, default="cfg/dataset.yaml")
    parser.add_argument("--model", type=str, default="cfg/model.yaml")
    parser.add_argument("--training", type=str, default="cfg/training.yaml")
    args = parser.parse_args()

    sidekit.nnet.xvector.xtrain(
        dataset_description=args.dataset,
        model_description=args.model,
        training_description=args.training,
    )


if __name__ == "__main__":
    main()
