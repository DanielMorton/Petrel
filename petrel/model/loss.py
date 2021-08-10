from . import AvgCounter


class LossCounter:
    """
    Stores counters for total loss, class loss and bounding box loss.
    """

    def __init__(self):
        self._loss = AvgCounter()
        self._class_loss = AvgCounter()
        self._box_loss = AvgCounter()

    def avg(self):
        """
        Returns the average value of the main loss counter.
        :return: The average value of the main loss counter.
        """
        return self._loss.avg

    def box_avg(self):
        """
        Return the average value of the bounding box loss counter.
        :return: The average value of the bounding box loss counter.
        """
        return self._box_loss.avg

    def box_loss(self):
        """
        Returns the bounding box loss counter.
        :return: The bounding box loss counter.
        """
        return self._box_loss

    def class_avg(self):
        """
        Returns the average value of the class loss counter.
        :return: The average value of the class loss counter.
        """
        return self._class_loss.avg

    def class_loss(self):
        """
        Returns the class loss counter.
        :return: The class loss counter
        """
        return self._class_loss

    def loss(self):
        """
        Returns the total loss counter.
        :return: The total loss counter.
        """
        return self._loss

    def update(self, output, batch_size=1, avg=False):
        """
        Updates the loss counters after a forward pass of a batch of data.
        :param output: The output of the forwrd pass.
        :param batch_size: The batch size of the forward pass.
        :param avg: Are the values in output averages or sums. Defaults to False for sums.
        """
        if avg:
            self._loss.update_avg(output["loss"].detach().item(), batch_size)
            self._class_loss.update_avg(output["class_loss"].detach().item(), batch_size)
            self._box_loss.update_avg(output["box_loss"].detach().item(), batch_size)
        else:
            self._loss.update(output["loss"].detach().item(), batch_size)
            self._class_loss.update(output["class_loss"].detach().item(), batch_size)
            self._box_loss.update(output["box_loss"].detach().item(), batch_size)
