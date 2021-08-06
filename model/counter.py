class AvgCounter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.current = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, avg=False):
        self.current = val
        self.sum += val * n if avg else val
        self.count += n
        self.avg = self.sum / self.count

    def concat(self, other_meter):
        self.current += other_meter.current
        self.sum += other_meter.sum
        self.count += other_meter.count
        self.avg = self.sum / self.count


class LossCounter:
    def __init__(self, loss=None, class_loss=None, box_loss=None):
        self.loss = loss if loss else AvgCounter()
        self.class_loss = class_loss if class_loss else AvgCounter()
        self.box_loss = box_loss if box_loss else AvgCounter()

    def update(self, output, n=1, avg=False):
        self.loss.update(output["loss"].detach().item(), n, avg)
        self.class_loss.update(output["class_loss"].detach().item(), n, avg)
        self.box_loss.update(output["box_loss"].detach().item(), n, avg)

    def concat(self, other_counter):
        self.loss.concat(other_counter.loss)
        self.class_loss.concat(other_counter.class_loss)
        self.box_loss.concat(other_counter.box_loss)
