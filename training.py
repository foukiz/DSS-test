import torch
import wandb
import numpy as np

from tqdm import tqdm

from utils import unpack_batch




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
    inspect_gradients=False,
    display_every=None,
    display_epoch=False,
    use_wandb=False,
    track_norms=False,
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
    if validation:
        val_loader = torch.utils.data.DataLoader(
            dataset.val_ds,
            batch_size=batch_size,
            shuffle=False
        )

    train_size = len(dataset.train_ds)
    n_batches = train_size // batch_size + (train_size % batch_size != 0)

    if inspect_gradients:
        torch.autograd.set_detect_anomaly(True)

    for epoch in range(n_epochs):
        if not use_tqdm: print("entering epoch {}".format(epoch+1))
        if track_norms: model.initialize_layer_norms()

        # train mode -> gradients computing switched on
        model.train()

        # set the keys for training data we want to register, averaged over batches
        # / reinitialize them to 0
        stat_epoch = {"loss": 0., "lr": optimizer.param_groups[0]['lr']}
        if metrics:
            stat_epoch.update({name: 0. for name in metrics.keys()})

        if use_tqdm: train_loader = tqdm(train_loader)

        # enter the loop over batches
        for batch_idx, batch in enumerate(train_loader):
            batch_x, batch_y, batch_lengths = unpack_batch(batch, torch_device)
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
            stat_batch = training_step(batch_x, batch_y, batch_lengths, model, optimizer, loss_fn, metrics=metrics, get_gradients=get_gradients, torch_device=torch_device)

            # update training data: average loss, metrics etc
            stat_epoch = batch_update(stat_epoch, stat_batch, batch_idx)

            # display additional data
            if display_every and (batch_idx % display_every == 0):
                display_train_data(loss=stat_batch['loss'], batch_idx=batch_idx, n_batches=n_batches, epoch=epoch)

        if validation:
            stat_val = evaluate(val_loader, model, loss_fn, metrics=metrics, kind='validation', torch_device=torch_device)

        if scheduler is not None:
            if not validation: stat_val = None
            scheduler_update(scheduler, stat_val=stat_val)
            stat_epoch["lr"] = optimizer.param_groups[0]['lr']

        if display_epoch:
            display_train_data(epoch_loss=stat_epoch['loss'], **stat_val)

        if track_norms:
            model.average_layer_norms(n_batches=n_batches)

        if use_wandb:
            # TODO mettre une option pour customiser les données qu'on veut envoyer sur wandb
            wandb_dic = stat_epoch.copy()
            if validation:
                wandb_dic.update(stat_val)
            if hasattr(dataset, "naive_baseline"):
                wandb_dic["CCE baseline"] = dataset.naive_baseline
            if track_norms:
                #norms = model.compute_norms(L=dataset.seq_length)
                norms = model.compute_norms()
                layer_norms = {'layer_norm/'+k: v for k, v in model.layer_norms.items()}
                wandb_dic.update(norms)
                wandb_dic.update(layer_norms)
            #wandb.log(wandb_dic)
            try:
                wandb.log(wandb_dic)
            except Exception as e:
                print(f"WANDB ERROR: {e}")

    return model

def training_step(batch_x, batch_y, batch_lengths, model, optimizer, loss_fn, metrics, get_gradients=False, torch_device=None, **kwargs):

    model.train()
    batch_x, batch_y = batch_x.to(torch_device), batch_y.to(torch_device).view(-1)
    predictions = model(batch_x, batch_lengths).view(-1, model.output_size).squeeze()
    loss = loss_fn(predictions, batch_y)

    # update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    stat_batch = {"loss": loss.item()}

    if metrics:
        metrics_values = compute_metrics(metrics, predictions, batch_y, torch_device=torch_device)
        stat_batch.update(metrics_values)

    if get_gradients:
        grads = model.compute_gradients()
        stat_batch.update(grads)

    return stat_batch


def batch_update(stat_epoch, stat_batch, batch_idx):
    
    for k in stat_batch.keys():
        if k not in stat_epoch.keys(): stat_epoch[k] = 0.
        # update moving average
        stat_epoch[k] = (batch_idx * stat_epoch[k] + stat_batch[k]) / (batch_idx + 1)
    return stat_epoch


def evaluate(loader, model, loss_fn, metrics=None, kind='validation', torch_device=None, **kwargs):
    
    # TODO keep the possibility to have batch_size different from the size
    # of the whole validation / test dataset ?

    if kind == 'validation': prefix = 'val_'
    elif kind == 'test': prefix = 'test_'
    else: raise AttributeError("evaluation kind {} unknown".format(kind))

    model.eval()
    running_vloss = 0.
    n_batches = len(loader)
    if metrics: metric_values = {(prefix+name): 0. for name in metrics.keys()}

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, val_batch in enumerate(loader):
            vinputs, vlabels, vlengths = unpack_batch(val_batch, torch_device)
            vinputs = vinputs.to(torch_device)
            vlabels = vlabels.to(torch_device).view(-1)

            voutputs = model(vinputs, vlengths).reshape(-1, model.output_size)

            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

            if metrics:
                metric_batch = compute_metrics(metrics, voutputs, vlabels, torch_device=torch_device)
                for k in metric_batch.keys():
                    metric_values[prefix+k] += metric_batch[k]
                
    val_loss = running_vloss / n_batches
    stat_eval = {prefix+"loss": val_loss}
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



def display_train_data(round=3, **stats):
    display_str = ""
    for k, v in stats.items():
        display_str += "\t  {}: {}".format(k, np.round(v, round))
    print(display_str)



def scheduler_update(scheduler, stat_val=None):
    if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
        scheduler.step(stat_val['val_loss'])
    else:
        scheduler.step()