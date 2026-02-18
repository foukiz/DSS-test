import torch
import wandb
import numpy as np

from tqdm import tqdm
from torchmetrics import Accuracy




def train(
        model,
        dataset,
        n_epochs,
        batch_size,
        loss_fn,
        optimizer,
        metrics=None,
        scheduler=None,
        get_gradients=False,
        display_every=None,
        display_epoch=False,
        use_wandb=False,
        torch_device=None,
        use_tqdm=True,
        **kwargs
    ):

    train_loader = torch.utils.data.DataLoader(
        dataset.train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    validation = hasattr(dataset, "val_ds") and (dataset.val_ds is not None) and (len(dataset.val_ds) > 0)
    train_size = len(dataset.train_ds)
    n_batches = train_size // batch_size

    for epoch in range(n_epochs):
        if not use_tqdm: print("entering epoch {}".format(epoch+1))

        # train mode -> gradients computing switched on
        model.train()

        # set the keys for training data we want to register, averaged over batches
        # / reinitialize them to 0
        stat_epoch = {"loss": 0.}
        if metrics:
            stat_epoch.update({name: 0. for name in metrics.keys()})

        if use_tqdm: train_loader = tqdm(train_loader)

        # enter the loop over batches
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # set data to be displayed next to the progress bar
            if use_tqdm:
                train_loader.set_description(f"Epoch {epoch+1}/{n_epochs}")
                tqdm_postfix = {"avg loss": stat_epoch["loss"], "lr": optimizer.param_groups[0]['lr']}
                if metrics and 'accuracy' in metrics.keys():
                    tqdm_postfix["avg accuracy"] = stat_epoch["accuracy"]
                if hasattr(dataset, "naive_baseline"):
                    tqdm_postfix["baseline error"] = dataset.naive_baseline
                train_loader.set_postfix(tqdm_postfix)

            # make a training step, and record additional data if required
            stat_batch = training_step(batch_x, batch_y, model, optimizer, loss_fn, metrics=metrics, get_gradients=get_gradients, torch_device=torch_device)

            # update training data: average loss, metrics etc
            stat_epoch = batch_update(stat_epoch, stat_batch, batch_idx)

            # display additional data
            if display_every and (batch_idx % display_every == 0):
                display_train_data(loss=stat_batch['loss'], batch_idx=batch_idx, n_batches=n_batches, epoch=epoch)

        if validation:
            val_batch_size = batch_size
            stat_val = evaluate(dataset.val_ds, val_batch_size, model, loss_fn, metrics=metrics, kind='validation', torch_device=torch_device)

        if scheduler is not None:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(stat_val['val_loss'])
            else:
                scheduler.step()

        if display_epoch:
            display_train_data(epoch_loss=stat_epoch['loss'], **stat_val)

        if use_wandb:
            # TODO mettre une option pour customiser les données qu'on veut envoyer sur wandb
            wandb_dic = stat_epoch.copy()
            if validation:
                wandb_dic.update(stat_val)
            if hasattr(dataset, "naive_baseline"):
                wandb_dic["CCE baseline"] = dataset.naive_baseline
            wandb.log(wandb_dic)

    return model

def training_step(batch_x, batch_y, model, optimizer, loss_fn, metrics, torch_device=None, **kwargs):

    model.train()
    batch_x, batch_y = batch_x.to(torch_device), batch_y.to(torch_device).view(-1)
    predictions = model(batch_x).view(-1, model.output_size).squeeze()
    loss = loss_fn(predictions, batch_y)

    # update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #model.apply_weight_constraints_()

    stat_batch = {"loss": loss.item()}

    if metrics:
        metrics_values = compute_metrics(metrics, predictions, batch_y, torch_device=torch_device)
        stat_batch.update(metrics_values)

    return stat_batch


def batch_update(stat_epoch, stat_batch, batch_idx):
    for k in stat_batch.keys():
        if k not in stat_epoch.keys(): raise KeyError("key {} is not an epoch statistic".format(k))
        stat_epoch[k] = (batch_idx * stat_epoch[k] + stat_batch[k]) / (batch_idx + 1)
    return stat_epoch


def evaluate(dataset, batch_size, model, loss_fn, metrics=None, kind='validation', torch_device=None, **kwargs):
    
    # TODO keep the possibility to have batch_size different from the size
    # of the whole validation / test dataset ?

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if kind == 'validation': prefix = 'val_'
    elif kind == 'test': prefix = 'test_'
    else: raise AttributeError("evaluation kind {} unknown".format(kind))

    model.eval()
    running_vloss = 0.
    if metrics: metric_values = {(prefix+name): 0. for name in metrics.keys()}

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(torch_device), vlabels.to(torch_device).view(-1)
            voutputs = model(vinputs).view(-1, model.output_size).squeeze()
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            if metrics:
                metric_batch = compute_metrics(metrics, voutputs, vlabels, torch_device=torch_device)
                for k in metric_batch.keys():
                    metric_values[prefix+k] += metric_batch[k]
                
    val_loss = running_vloss / (i + 1)
    stat_eval = {prefix+"loss": val_loss.item()}
    if metrics: stat_eval.update({k:(v / (i+1)) for k, v in metric_values.items()})
    return stat_eval

def compute_metrics(metrics, batch_preds, batch_y, torch_device=None):
    """ metrics should be a dictionnary of the form {name: metric}, with name being a string
        and metric a torchmetrics.metric instance
    """

    metrics_values = {}

    for name, metric in metrics.items():
        metric.to(torch_device)
        batch_preds, batch_y = batch_preds.to(torch_device), batch_y.to(torch_device)
        metrics_values[name] = metric(batch_preds, batch_y).item()

    return metrics_values


def get_layer_gradients(layer, layer_name='', operation='average_norm'):
    gradients = {}
    for name, p in layer._parameters.items():
        if p is None: continue
        grad = p.grad
        if grad is None: grad = torch.tensor(0., dtype=p.dtype)
        if operation == 'average_norm':
            gradients['gradients/' + layer_name + '_' +name] = torch.mean(torch.abs(grad))
        else: raise ValueError("operation {} is unknown. Allowed operations are \"average_norm\"".format(operation))
        # TODO autres opérations éventuelles
    return gradients


def display_train_data(round=3, **stats):
    display_str = ""
    for k, v in stats.items():
        display_str += "\t  {}: {}".format(k, np.round(v, round))
    print(display_str)