from utils import find_file, iter_configs, assert_single_run_config, has_internet

import os
os.environ["WANDB_MODE"] = "online" if has_internet() else "offline"

import torch
import random
import numpy as np
import os

import argparse
import wandb

import time
from datetime import datetime, timedelta
import yaml

from typing import Optional

from training import train, evaluate
from config import Config
from models import DSS, S4, TransformerEncoder

from datasets import copy_task, listops, seq_cifar10, imdb, aan, pathfinder, smnist, pmnist, ptb




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="conf/copytask/copy_task_dss.yaml"
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--pre_seed", action="store_true")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--save_network", action="store_true")
    parser.add_argument("--save_name", type=str, default=None)
    return parser.parse_args()


ARGS = vars(parse_args())


def make_model(name, **kwargs):
    low_name = name.lower()
    models = {'dss': DSS, 's4': S4, 'transformer': TransformerEncoder}
    if low_name not in models:
        err_str = "{} is not a correct model name, accepted models are".format(low_name)
        for i, k in enumerate(models.keys()):
            if i == len(models) - 1 and i > 0: err_str += " and {}".format(k)
            else: err_str += " {},".format(k)
        err_str += " (case does not matter)"
        raise KeyError(err_str)

    model = models[low_name]
    return model(**kwargs)


def make_dataset(name, **kwargs):
    low_name = name.lower()
    datasets = {
        'copymemory': copy_task.CopyMemory,
        'listops': listops.ListOps,
        'scifar10': seq_cifar10.sCIFAR10,
        'imdb': imdb.IMDB,
        'aan': aan.AAN,
        'pathfinder': pathfinder.Pathfinder,
        'smnist': smnist.sMNIST,
        'pmnist': pmnist.pMNIST,
        'ptb': ptb.PennTreebank
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


def save_model(model, dataset, save_name, stat_dict=None, **kwargs):
    os.makedirs("models", exist_ok=True)
    torch.save(model, f'models/{save_name}.pth')
    # print in file performance
    if stat_dict:
        try:
            with open(f'models/{save_name}.txt', 'w') as f:
                for kk, vv in stat_dict.items():
                    f.write(f"{kk}: {vv}\n")

                if kwargs.get('run_url'):
                    f.write(f"\nWandb run URL:\n{kwargs['run_url']}\n\n")

                f.write(f"\n\n================= dataset =================\n\n{str(dataset)}\n\n")
                f.write(f"\n\n================== model ==================\n\n{str(model)}")
        except FileNotFoundError:
            print(f"Warning: could not save performance stats for model {save_name}, file not found")
            pass






def launch(
    config=None,
    use_wandb=False,
    use_tqdm=True,
    pre_seed=False,
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
        
    if ARGS['device'] == 'cuda' or cfg.train['torch_device'].startswith('cuda'):
        if torch.cuda.is_available():
            device = 'cuda'
            cfg.train['torch_device'] = device
        else:
            print("Warning: cuda device specified but not available, using cpu instead")
            device = 'cpu'

    if pre_seed:
        seed_everything(cfg.model['seed'], workers=True)
        cfg.model.pop('seed', None)

    kwargs = {}
    dataset = make_dataset(**cfg.dataset)
    if hasattr(dataset, 'padding_idx'):
        kwargs.update({'padding_idx': dataset.padding_idx})
        if 'crossentropyloss' in config['TRAIN']['LOSS_FN'].lower():
            cfg.train['loss_fn'] = torch.nn.CrossEntropyLoss(ignore_index=dataset.padding_idx)
    input_dim = dataset.input_flat_dimension
    output_dim = dataset.num_outputs
    try:
        if config['TRAIN']['TRACK_NORMS'] is True: kwargs['track_norms'] = True
    except KeyError:
        pass
    model = make_model(data_dim=input_dim, output_size=output_dim, **cfg.model, **kwargs).to(device)

    print(f'\n{model}\n')

    cfg.instantiate_optimizer(params=model.parameters())
    cfg.instantiate_scheduler()

    if training:
        print("\n=== Launching training ===\n")
        start = time.time()
        model = train(model, dataset, **cfg.train)
        elapsed = time.time() - start
        print(f"\nTraining time: {timedelta(seconds=int(elapsed))}\n")

    if dataset.test_ds:
        test_batch_size = cfg.train['batch_size']
        loader = torch.utils.data.DataLoader(dataset.test_ds, test_batch_size, shuffle=False)
        stat_test = evaluate(
            loader,
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
        run_url = wandb.run.url if use_wandb else None
        save_model(model, dataset, save_name, stat_dict=stat_test, run_url=run_url)

    if use_wandb:
        wandb.log(stat_test)
        for kk, vv in stat_test.items():
            wandb.run.summary["final test evaluation/"+kk] = vv

        wandb.finish()

    print("\n\n")

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



def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    r"""Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.
    In addition, sets the following environment variables:

    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - ``PL_SEED_WORKERS``: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If ``None``, it will read the seed from ``PL_GLOBAL_SEED`` env variable. If ``None`` and the
            ``PL_GLOBAL_SEED`` env variable is not set, then the seed defaults to 0. If seed is
            not in bounds or cannot be cast to int, a ValueError is raised.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning_fabric.utilities.seed.pl_worker_init_function`.
        verbose: Whether to print a message on each rank with the seed being set.

    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                raise ValueError(f"Invalid seed specified via PL_GLOBAL_SEED: {repr(env_seed)}")
    elif not isinstance(seed, int):
        seed = int(seed)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed




if __name__ == "__main__":
    multiple_launch(**ARGS)
