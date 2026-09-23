from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .trainer import Trainer


def apply_trained_task_network(network_path, data):
    """
    Load a trained TADRED network and apply it to a dataset.

    Parameters
    ----------
    network_path : str or Path
        Path to the trained PyTorch checkpoint.

    data : np.ndarray, dict, str or Path
        Input data, or path to a `.npy` file containing the input data.

        The data may contain either:

        - the full set of measurements used during TADRED training, with shape
          `(n_samples, n_full_measurements)`, or
        - only the measurements retained by the final TADRED mask, with shape
          `(n_samples, n_selected_measurements)`.

    Returns
    -------
    output : torch.Tensor
        Network output.
    """

    network_path = Path(network_path)

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        network_path,
        map_location="cpu",
        weights_only=False,
    )

    args = OmegaConf.create(checkpoint["args"])
    data_features_norm = checkpoint["data_features_norm"]

    # # Load trained network
    # checkpoint = torch.load(
    #     network_path,
    #     map_location="cpu",
    #     weights_only=False,
    # )

    # args = OmegaConf.create(checkpoint["args"])
    # data_features_norm = checkpoint["data_features_norm"]


    # ---------------------------------------------------------
    # Reconstruct trained TADRED model
    # ---------------------------------------------------------

    nnet = Trainer(
        tadred_train_eval=args.tadred_train_eval,
        network=args.network,
        data_features_norm=data_features_norm,
        train_pytorch=args.train_pytorch,
        other_options=args.other_options,
    )

    nnet.device = "cpu"
    nnet._create_model()

    nnet.model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    nnet.model.eval()


    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    if isinstance(data, (str, Path)):
        data = np.load(
            data,
            allow_pickle=True,
        )

    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(
            "Input data must have shape "
            "(n_samples, n_measurements)."
        )

    
    # ---------------------------------------------------------
    # Determine retained measurements
    # ---------------------------------------------------------

    m = nnet.model.downsampling_mult_layer.m

    selected_indices = torch.where(
        m > 0
    )[0].cpu().numpy()

    n_full_measurements = len(m)
    n_selected_measurements = len(selected_indices)

    n_input_measurements = data.shape[-1]


    # ---------------------------------------------------------
    # Expand reduced data if necessary
    # ---------------------------------------------------------

    if n_input_measurements == n_selected_measurements:

        data_full = np.zeros(
            (data.shape[0], n_full_measurements),
            dtype=np.float32,
        )

        data_full[:, selected_indices] = data

    elif n_input_measurements == n_full_measurements:

        data_full = data.astype(np.float32)

    else:

        raise ValueError(
            f"Input contains {n_input_measurements} measurements, but the "
            f"trained network expects either {n_selected_measurements} "
            f"selected measurements or {n_full_measurements} full "
            "measurements."
        )
    
    # Apply network
    with torch.no_grad():
        output = nnet.model.forward_eval(
            data_full,
            score=1,
        )

    return output
 