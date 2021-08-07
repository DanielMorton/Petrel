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
