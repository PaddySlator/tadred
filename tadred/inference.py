from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .trainer import Trainer



def apply_trained_task_network(network_path, data_path):
    """
    Load a trained TADRED network and apply it to a dataset.

    Parameters
    ----------
    network_path : str or Path
        Path to the trained PyTorch checkpoint.

    data_path : str or Path
        Path to the .npy dataset. The loaded dictionary is expected
        to contain a 'test' entry.

    Returns
    -------
    output : torch.Tensor
        Network output for the test data.
    """

    network_path = Path(network_path)
    data_path = Path(data_path)

    # Load trained network
    checkpoint = torch.load(
        network_path,
        map_location="cpu",
        weights_only=False,
    )

    args = OmegaConf.create(checkpoint["args"])
    data_features_norm = checkpoint["data_features_norm"]

    nnet = Trainer(
        tadred_train_eval=args.tadred_train_eval,
        network=args.network,
        data_features_norm=data_features_norm,
        train_pytorch=args.train_pytorch,
        other_options=args.other_options,
    )

    nnet.device = "cpu"
    nnet._create_model()

    nnet.model.load_state_dict(checkpoint["model_state_dict"])
    nnet.model.eval()

    # Load data
    dataset = np.load(
        data_path,
        allow_pickle=True,
    ).item()

    # Apply network
    with torch.no_grad():
        output = nnet.model.forward_eval(
            dataset["test"],
            score=1,
        )

    return output