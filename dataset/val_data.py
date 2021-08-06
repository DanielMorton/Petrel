import torch
from .petrel import PetrelDataset


class ValDataset(PetrelDataset):

    def __init__(self, meta_data,
                 boxes,
                 image_root,
                 transform=None,
                 train_pipe=False):
        super(ValDataset, self).__init__(meta_data, boxes, image_root, transform, train_pipe)

    def __getitem__(self, index):
        """Retrieves the image and boxes with the specified index."""
        image_meta = self.meta_data.loc[index]
        image_boxes = self.boxes[self.boxes["file"] == image_meta["file"]]
        image, bboxes, labels = self.load_image_and_boxes(image_meta, image_boxes)
        target = {"bboxes": bboxes,
                  "labels": labels}

        if self.transform:
            sample = self.transform(image=image,
                                    bboxes=target["bboxes"],
                                    labels=target["labels"])
            image, target["bboxes"] = sample['image'], self._box_to_tensor(sample)
            target["labels"] = torch.tensor(sample["labels"])
        return image, target
