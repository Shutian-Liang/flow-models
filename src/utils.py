import os
import argparse
import yaml
from prettytable import PrettyTable

def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VAE Training Script")
    
    # basic set
    parser.add_argument("--data_path", type=str, default="./cifar10/", help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--patch_size", type=int, default=[32, 32], help="Size of the image patches")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

    # model choices
    parser.add_argument("--model", type=str, default="simple_flow", choices=["simple_flow", 'ddpm'], help="Model type")
    args = parser.parse_args()
    return args

def count_parameters(model):
    """Count the number of parameters in the model
    Args:
        model: the model to count parameters
    Returns:
        total_params: the total number of parameters in the model
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def load_config(model_name, base_path="./configs/"):
    """
    Load configuration from a YAML file based on the model name.
    Args:
        model_name (str): Name of the model to load configuration for.
        base_path (str): Base path where the config files are stored.
    Returns:
        dict: Configuration parameters.
    """
    config_path = os.path.join(base_path, f"{model_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def checkandcreate(path):
    """
    check if the path exists, if not create it
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created directory {path}')
    else:
        print(f'Directory {path} already exists')
