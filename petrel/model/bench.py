from effdet import DetBenchTrain


class DetBenchTrainVal(DetBenchTrain):
    def __init__(self, model, config):
        super(DetBenchTrainVal, self).__init__(model, config)

    def forward(self, x, target):
        class_out, box_out = self.model(x)
        if self.anchor_labeler is None:
            # target should contain pre-computed anchor labels if labeler not present in bench
            assert 'label_num_positives' in target
            cls_targets = [target[f'label_cls_{l}'] for l in range(self.num_levels)]
            box_targets = [target[f'label_bbox_{l}'] for l in range(self.num_levels)]
            num_positives = target['label_num_positives']
        else:
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                target['bbox'], target['cls'])

        loss, class_loss, box_loss = self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
        output = {'loss': loss, 'class_loss': class_loss, 'box_loss': box_loss}
        return output
