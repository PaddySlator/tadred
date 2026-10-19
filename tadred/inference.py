from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .trainer import Trainer


def apply_trained_task_network(trained_model, data):
    """
    Apply a trained TADRED network to a dataset.

    Parameters
    ----------
    trained_model : torch.nn.Module or str or Path
        Live trained TADRED model or path to a saved checkpoint.

    data : np.ndarray, str or Path
        Input data, or path to a `.npy` file.

        Data may contain either the full set of measurements or only
        the measurements retained by the final TADRED mask.

    Returns
    -------
    output : torch.Tensor
        Network output.
    """

    # ---------------------------------------------------------
    # Load checkpoint or use live model
    # ---------------------------------------------------------

    if isinstance(trained_model, (str, Path)):

        checkpoint = torch.load(
            trained_model,
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

        nnet.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = nnet.model

    else:
        model = trained_model

    model.eval()

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

    m = model.downsampling_mult_layer.m

    selected_indices = torch.where(
        m > 0
    )[0].cpu().numpy()

    n_full_measurements = len(m)
    n_selected_measurements = len(selected_indices)

    n_input_measurements = data.shape[-1]

    # ---------------------------------------------------------
    # Expand reduced data if necessary
    # ---------------------------------------------------------

    if n_input_measurements == n_full_measurements:

        data_full = data.astype(np.float32)

    elif n_input_measurements == n_selected_measurements:

        data_full = np.zeros(
            (data.shape[0], n_full_measurements),
            dtype=np.float32,
        )

        data_full[:, selected_indices] = data

    else:

        raise ValueError(
            f"Input contains {n_input_measurements} measurements, but the "
            f"trained network expects either {n_selected_measurements} "
            f"selected measurements or {n_full_measurements} full "
            "measurements."
        )

    # ---------------------------------------------------------
    # Apply network
    # ---------------------------------------------------------

    device = next(model.parameters()).device

    data_full = torch.from_numpy(data_full).to(device)

    with torch.no_grad():
        output = model.forward_eval(
            data_full,
            score=1,
        )

    return output