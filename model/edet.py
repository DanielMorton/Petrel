import torch.optim
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict


def load_net(epoch,
             image_size,
             config_name,
             num_classes=1,
             max_det_per_image=1000,
             soft_nms=False,
             device=None,
             predict=True):
    config = get_efficientdet_config(config_name)
    if type(image_size) == int:
        config.image_size = [image_size, image_size]
    else:
        config.image_size = image_size
    config.max_det_per_image = max_det_per_image
    config.soft_nms = soft_nms
    config.num_classes = num_classes
    if epoch == 0:
        net = EfficientDet(config, pretrained_backbone=True)
    else:
        net = EfficientDet(config, pretrained_backbone=False)
    if predict:
        net = DetBenchPredict(net)
        net.eval()
    else:
        net = DetBenchTrain(net)
        net.train()
    if device:
        net = net.to(device)

    return net


def load_optimizer(opt_type,
                   model,
                   learning_rate,
                   **kwargs):
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



