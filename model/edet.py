from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict


def load_edet(config_name,
              image_size,
              pretrained_backbone=True,
              num_classes=1,
              max_det_per_image=1000,
              soft_nms=False,
              device=None,
              train=True):
    """
    Loads the EfficientDet model with the given config name, input size, and output classes.
    Can load models with or without pretrained weights. The latter case is for loading a model with custom weights


    :param config_name: Name of the mdoel to load.
    :param image_size: Size the input images.
    :param pretrained_backbone: Load pretrained weights. Defaults to true.
    :param num_classes: Number of prediction classes.
    :param max_det_per_image: Maximum number of detection regions to predict.
    :param soft_nms: Use soft non-max suppression. Defaults to False as soft nms is very slow.
    :param device: Device to load the model on. CPU or CUDA. Defaults to None, equivalent to CPU
    :param train: Is the model being trained. Defaults to True
    :return: EfficientDet model.
    """
    config = get_efficientdet_config(config_name)
    if type(image_size) == int:
        config.image_size = [image_size, image_size]
    else:
        config.image_size = image_size
    config.max_det_per_image = max_det_per_image
    config.soft_nms = soft_nms
    config.num_classes = num_classes
    model = EfficientDet(config, pretrained_backbone=pretrained_backbone)

    if train:
        model = DetBenchTrain(model)
        model.train()
    else:
        model = DetBenchPredict(model)
        model.eval()
    if device:
        model = model.to(device)

    return model
