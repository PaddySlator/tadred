"""
Copyright 2024 Stefano B. Blumberg

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import pickle as pkl
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from hydra import compose, initialize
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def create_out_dirs(
    out_base: str | None = None,
    proj_name: str = "results",
    run_name: str = "tadred_output_",
) -> dict[str, Path | None]:
    """Create directories to save output if out_base is not None

    Output saved in <out_base>/<proj_name>/<run_name>/
    Results saved in <out_base>/results/<run_name>_all.npy
    """
    # if out_base is not None and len(proj_name) is not None:
    if out_base is not None:
        out_base_dir = Path(out_base, proj_name)
        out_base_dir.mkdir(parents=True, exist_ok=True)
        log.info("Output base directory: %s", out_base_dir)
        results_dir = Path(out_base_dir, "full_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        log.info("Output results directory: %s", out_base_dir)
        results_fn = Path(results_dir, run_name + "full_results.npy")

        save_model_path = Path(out_base_dir, run_name)
        log.info("Model is saved to %s", save_model_path)
    else:
        out_base_dir = None
        results_fn = None
        save_model_path = None
        log.info("Did not create output base directory")
        log.info("Did not create model saved dir")

    out_dirs = dict(
        out_base_dir=out_base_dir,
        results_fn=results_fn,
        save_model_path=save_model_path,
    )
    return out_dirs


def load_yaml(file_path: str):
    """Loads .yaml config file from file_path on disk"""
    with open(file_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def load_base_args() -> DictConfig:
    with initialize(version_base=None, config_path="."):
        args = compose(config_name="cfg")
    return args


def load_base_args_combine_with_yaml(path: str) -> DictConfig:
    args = load_base_args()
    loaded_yaml = load_yaml(path)
    args.merge_with(loaded_yaml)
    return args


def load_results(
    full_path: str | None = None,
    out_base_dir: str | None = None,
    run_name: str | None = None,
):
    """Load results file from TADRED save.

    Option (i) Pass full path link
    Option (ii) Pass out_base_dir and run_name
    """
    # TODO use truthyness
    if full_path is not None:
        load_path = Path(full_path)
    elif out_base_dir is not None and run_name is not None:
        load_path = Path(out_base_dir, "results", f"{run_name}_tadred_full_results.npy")
    else:
        raise ValueError("Pass either full_path, or out_base_dir and run_name to create full_path")
    results_load = np.load(str(load_path), allow_pickle=True).item()
    return results_load


def save_results_dir(
    out_base_dir: Path | None,
    results_fn: Path | None,
    results: dict,
) -> None:
    """Save final results file if requested"""
    if out_base_dir is not None:
        log.info("Saving final results in %s", results_fn)
        with open(str(results_fn), "wb") as file:
            pkl.dump(results, file)
        # np.save(str(results_fn), results)
    else:
        log.info("Do not save final results")


def set_random_seed(seed: int) -> None:
    """Set random seed"""

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # for cudnn backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = Falserr
    # os.environ["PYTHONHASHSEED"] = str(seed)

    log.info(f"Random seed is {seed}")


def format_runtime(time_s):
    """Format a duration in seconds as hours, minutes and seconds."""
    hours, remainder = divmod(time_s, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
    if minutes:
        return f"{int(minutes)}m {seconds:.1f}s"
    return f"{seconds:.1f}s"
