import torch


def load_optimizer(opt_type,
                   model,
                   learning_rate,
                   **kwargs):
    """
    Returns the optimizer for model training.
    :param opt_type: Returns the
    :param model: The model to be trained.
    :param learning_rate: Initial learning rate.
    :param kwargs:
    :return:
    """
    if opt_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(),
                                lr=learning_rate)
    if opt_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=kwargs.get("weight_decay", 4e-5))
    if opt_type.lower() == "rms":
        return torch.optim.RMSprop(model.parameters(),
                                   lr=learning_rate,
                                   momentum=kwargs.get("momentum", 0.9))
    if opt_type.lower() == "sgd":
        return torch.optim.SGD(model.parameters(),
                               lr=learning_rate,
                               momentum=kwargs.get("momentum", 0.9))
