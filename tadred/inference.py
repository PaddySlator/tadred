from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .trainer import Trainer



def apply_trained_task_network(trained_model, data):
    """
    Apply a trained TADRED task network.

    trained_model can be either:
        - a live torch.nn.Module
        - a path to a saved checkpoint
    """

    # --------------------------------------------------
    # Load model if a checkpoint path was supplied
    # --------------------------------------------------

    if isinstance(trained_model, (str, Path)):
            # Load trained network
            checkpoint = torch.load(
                trained_model,
                map_location="cpu",
                weights_only=False,
            )
        
            args = OmegaConf.create(checkpoint["args"])
            data_features_norm = checkpoint["data_features_norm"]
        
        
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
            
            model = nnet.model           
    else:
        model = trained_model
    
    model.eval()

    # --------------------------------------------------
    # Load data if a path was supplied
    # --------------------------------------------------
    
    if isinstance(data, (str, Path)):
        dataset = np.load(
            data,
            allow_pickle=True,
        )
    else:
        dataset = np.asarray(data)
    
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

    if n_input_measurements == n_selected_measurements:

        data_full = np.zeros(
            (dataset.shape[0], n_full_measurements),
            dtype=np.float32,
        )

        data_full[:, selected_indices] = dataset

    elif n_input_measurements == n_full_measurements:

        data_full = dataset.astype(np.float32)

    else:

        raise ValueError(
            f"Input contains {n_input_measurements} measurements, but the "
            f"trained network expects either {n_selected_measurements} "
            f"selected measurements or {n_full_measurements} full "
            "measurements."
        )
        
    # ---------------------------------------------------------
    # Apply Neural Network to the full data
    # ---------------------------------------------------------
    with torch.no_grad():
        output = model.forward_eval(
            data_full,
            score=1,
        )

    return output