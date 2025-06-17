##
## EPITECH PROJECT, 2025
## root [SSH: robocar-scaleway]
## File description:
## utils
##

import yaml
import os

def load_pad_divisor_from_run_dir(run_dir: str) -> int:
    """
    Load the padding divisor from the run directory.

    Args:
        run_dir (str): Path to the run directory.

    Returns:
        int: The padding divisor.
    """
    metadata_path = f"{run_dir}/metadata.yaml"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    with open(metadata_path, 'r') as file:
        metadata = yaml.safe_load(file)
    pad_divisor = metadata["pad_divisor"]
    if not isinstance(pad_divisor, int):
        raise ValueError(f"Expected pad_divisor to be an integer, got {type(pad_divisor)}")
    return pad_divisor
