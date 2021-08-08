import torch


def load_scheduler(scheduler_type,
                   optimizer,
                   **kwargs):
    if scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                          T_max=kwargs.get("T_max", 100),
                                                          eta_min=kwargs.get("eta_min", 0),
                                                          verbose=True)
    elif scheduler_type.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=kwargs.get("gamma", 1.0),
                                                      verbose=True)
    elif scheduler_type.lower() == "mult":
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                                         lr_lambda=kwargs.get("lr_lambda"),
                                                         verbose=True)
