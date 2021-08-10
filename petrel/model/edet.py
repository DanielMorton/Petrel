import torch
from effdet import get_efficientdet_config, create_model_from_config


def load_edet(config_name,
              image_size,
              num_classes=1,
              max_det_per_image=1000,
              soft_nms=False,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              train=True):
    """
    Loads the EfficientDet model with the given config name, input size, and output classes.
    Can load models with or without pretrained weights. The latter case is for loading a model with custom weights


    :param config_name: Name of the mdoel to load.
    :param image_size: Size the input images.
    :param num_classes: Number of prediction classes.
    :param max_det_per_image: Maximum number of detection regions to predict.
    :param soft_nms: Use soft non-max suppression. Defaults to False as soft nms is very slow.
    :param device: Device to load the model on. CPU or CUDA. Defaults using CUDA if available, otherwise CPU
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
    model = create_model_from_config(config, pretrained=True,
                                     bench_task="train" if train else "predict",
                                     max_det_per_image=max_det_per_image,
                                     num_classes=num_classes,
                                     soft_nms=soft_nms)
    if device:
        model = model.to(device)

    return model
