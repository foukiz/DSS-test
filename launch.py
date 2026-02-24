from utils import find_file, iter_configs, assert_single_run_config, has_internet

import os
os.environ["WANDB_MODE"] = "online" if has_internet() else "offline"

import torch

import argparse
import wandb

from datetime import datetime
import yaml

from training import train, evaluate
from config import Config
from models import *

from datasets import copy_task, listops, seq_cifar10




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="conf/copytask/copy_task_dss.yaml"
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--save_network", action="store_true")
    parser.add_argument("--save_name", type=str, default=None)
    return parser.parse_args()


ARGS = vars(parse_args())


def make_model(name, **kwargs):
    low_name = name.lower()
    models = {'dss': DSS}
    if low_name not in models:
        err_str = "{} is not a correct model name, accepted models are".format(low_name)
        for i, k in enumerate(models.keys()):
            if i == len(models) - 1 and i > 0: err_str += " and {}".format(k)
            else: err_str += " {},".format(k)
        err_str += " (case doeslow_ not matter)"
        raise KeyError(err_str)

    model = models[low_name]
    return model(**kwargs)


def make_dataset(name, **kwargs):
    low_name = name.lower()
    datasets = {
        'copymemory': copy_task.CopyMemory,
        'listops': listops.ListOps,
        'scifar10': seq_cifar10.sCIFAR10
    }
    if low_name not in datasets:
        err_str = "{} is not a correct dataset name, accepted datasets are".format(low_name)
        for i, k in enumerate(datasets.keys()):
            if i == len(datasets) - 1 and i > 0: err_str += " and {}".format(k)
            else: err_str += " {},".format(k)
        err_str += " (case does not matter)"
        raise KeyError(err_str)
    
    dataset = datasets[low_name]
    return dataset(**kwargs)


def save_model(model, dataset, save_name, stat_dict=None):
    os.makedirs("models", exist_ok=True)
    torch.save(model, f'models/{save_name}.pth')
    # print in file performance
    if stat_dict:
        try:
            with open(f'models/{save_name}.txt', 'w') as f:
                for kk, vv in stat_dict.items():
                    f.write(f"{kk}: {vv}\n")
                f.write(f"\n\n================= dataset =================\n\n{str(dataset)}\n\n")
                f.write(f"\n\n================== model ==================\n\n{str(model)}")
        except FileNotFoundError:
            print(f"Warning: could not save performance stats for model {save_name}, file not found")
            pass






def launch(
        config=None,
        use_wandb=False,
        use_tqdm=True,
        device='cpu',
        save_network=False,
        save_name=None,
        **kwargs
    ):
    """ Launch a single experiment, including setting up the model and the dataset,
        and training the model on the dataset, with evaluation and testing.
    """

    try:
        # if config is a config file - e.g. yaml
        conf_file = find_file(config)
        cfg = Config(conf_file=conf_file)
    except TypeError:
        # if config is a config dictionnary
        assert_single_run_config(config)
        cfg = Config(conf_dic=config)

    project_name = cfg.project
    use_wandb = use_wandb or cfg.train['use_wandb']
    cfg.train['use_wandb'] = use_wandb
    use_tqdm = use_tqdm or cfg.train['use_tqdm']
    cfg.train['use_tqdm'] = use_tqdm

    training = not ARGS['no_train']

    if use_wandb:
        wandb.init(
                # set the wandb project where this run will be logged
                project=project_name,  # "projunn_quantized",
                # track hyperparameters and run metadata
                config=cfg.config,
            )
        
    if ARGS['device'] == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
            cfg.train['torch_device'] = device
        else:
            print("Warning: cuda device specified but not available, using cpu instead")
            device = 'cpu'

    dataset = make_dataset(**cfg.dataset)
    input_dim = dataset.input_flat_dimension
    output_dim = dataset.num_outputs
    model = make_model(data_dim=input_dim, output_size=output_dim, **cfg.model).to(device)

    cfg.instantiate_optimizer(params=model.parameters())
    cfg.instantiate_scheduler()

    if training:
        print("\n=== Launching training ===\n")
        model = train(model, dataset, **cfg.train)

    if dataset.test_ds:
        test_batch_size = cfg.train['batch_size']
        stat_test = evaluate(
            dataset.test_ds,
            test_batch_size,
            model,
            loss_fn=cfg.train['loss_fn'],
            metrics=cfg.train['metrics'],
            kind='test',
            torch_device=cfg.train['torch_device']
        )

    if save_network:
        if save_name is None:
            save_name = f"{cfg.model['name']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if dataset.test_ds is None: stat_test=None
        save_model(model, dataset, save_name, stat_dict=stat_test)

    if use_wandb:
        wandb.log(stat_test)
        for kk, vv in stat_test.items():
            wandb.run.summary["final test evaluation/"+kk] = vv

        wandb.finish()

    return model




def multiple_launch(config=None, **kwargs):
    """ Launch experiments based on a grid search defined in the config arg.
        An iterator object is built out of config, containing all the single-run
        configs, over which single experiments are performed one at a time.
    """

    if not isinstance(config, str):
        raise TypeError("config should be a string (a config file path), not {}".format(config))
    conf_file = find_file(config)
    with open(conf_file, 'r') as f:
        c = yaml.safe_load(f)
    conf_iterator = iter_configs(c)
    for cfg in conf_iterator:
        launch(config=cfg, **kwargs)





if __name__ == "__main__":
    multiple_launch(**ARGS)
