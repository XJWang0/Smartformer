import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

# from binarized_notmnist_dataset import NotMNISTDataset
from PIL import Image
from settings import *
from torch import nn
from torch.utils.data import DataLoader
from Smartformer import init_datasets# , init_model
from deformer import CP_DEformer_, DEformer, DEformer_

def init_model(opts, train_dataset):
    model_config = opts["model"]
    model_config["img_size"] = train_dataset.img_size
    model_config["pos_in_feats"] = 2 if opts["train"]["dataset"] == "mnist" else 3
    model_config["pixel_n"] = 1 if opts["train"]["dataset"] == "mnist" else 256
    # model = CP_DEformer_(**model_config)
    model = DEformer_(**model_config)
    return model

def get_multi_order_nlls():
    JOB = "2"
    EXPERIMENTS_DIR = './Deformer_experiments'
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize datasets.
    (train_dataset, _, _, _, _, test_loader) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    orders = 10
    n_pix = 28 ** 2
    pix_idxs = np.arange(n_pix)
    np.random.seed(2010)
    test_nlls = []
    for order in range(orders):
        print(order, flush=True)
        np.random.shuffle(pix_idxs)
        with torch.no_grad():
            for (test_idx, test_tensors) in enumerate(test_loader):
                test_tensors["pixels"] = test_tensors["pixels"][pix_idxs]
                test_tensors["positions"] = test_tensors["positions"][pix_idxs]

                preds = model(test_tensors)
                preds = preds.flatten().to(device)
                # preds = preds.view(preds.size(0), preds.size(1))


                labels = test_tensors["pixels"].flatten().to(device)
                # labels = test_tensors["pixels"].to(device)

                loss = criterion(preds, labels)
                test_nlls.append(n_pix * loss.item())

    print(sum(test_nlls) / len(test_nlls))
    fg = sns.displot(test_nlls)
    plt.xlabel("NLL")
    plt.tight_layout()
    # fg.fig.savefig("D:/ftp/Wxj/Subject/deformer/Deformer_experiments/deformer_test_nlls_hist.png")
    fg.fig.savefig("./Deformer_experiments/deformer_test_nlls_hist2.png")


if __name__ == "__main__":
    get_multi_order_nlls()