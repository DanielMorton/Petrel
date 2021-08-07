import torch
from .petrel import PetrelDataset


class TrainDataset(PetrelDataset):

    def __init__(self, meta_data,
                 boxes,
                 image_root,
                 transform=None,
                 default_transform=None,
                 max_iter=100):
        super(TrainDataset, self).__init__(meta_data, boxes, image_root, transform, train_pipe=True)
        self.default_transform = default_transform if default_transform else transform
        self.max_iter = max_iter

    def __getitem__(self, index: int):
        """Retrieves the image and boxes with the specified index."""
        image_meta = self.meta_data.loc[index]
        image_boxes = self.boxes[self.boxes["file"] == image_meta["file"]]
        image, bboxes, labels = self.load_image_and_boxes(image_meta, image_boxes)
        target = {"bboxes": bboxes,
                  "labels": labels}
        if self.transform and bboxes.shape[0] > 0:
            for i in range(self.max_iter):
                sample = self.transform(image=image,
                                        bboxes=target["bboxes"],
                                        labels=target["labels"])
                if len(sample["bboxes"]) > 0:
                    image, target["bboxes"] = sample["image"], self._box_to_tensor(sample["bboxes"])
                    target["labels"] = torch.tensor(sample["labels"])
                    return image, target
        if self.default_transform:
            sample = self.transform(image=image,
                                    bboxes=target["bboxes"],
                                    labels=target["labels"])
            image, target["bboxes"] = sample["image"], self._box_to_tensor(sample["bboxes"])
            target["labels"] = torch.tensor(sample["labels"])
            return image, target
