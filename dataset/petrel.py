from abc import ABC

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class PetrelDataset(Dataset, ABC):
    def __init__(self, meta_data,
                 boxes,
                 image_root,
                 transform=None,
                 train_pipe=False):
        super(PetrelDataset).__init__()

        self.meta_data = meta_data
        self.boxes = boxes
        self.image_root = image_root
        self.transform = transform
        self.train_pipe = train_pipe

    def _box_to_tensor(self, sample):
        """Convert boundind box array to tensor"""
        if len(sample["bboxes"]) > 0:
            bboxes = torch.tensor(sample["bboxes"])
        else:
            bboxes = torch.zeros((0, 4))
        # Convert bounded box to yxyx format if in training pipline
        if self.train_pipe:
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
        return bboxes

    def __len__(self) -> int:
        """Returns the number of images."""
        return self.meta_data.shape[0]

    def load_image_and_boxes(self, image_meta, image_boxes):
        """Loads image corresponding to image_meta row.
               Converts bounding boxes to x_min, y_min, x_max, y_max format.
            """
        image = cv2.cvtColor(
            cv2.imread(f"{self.image_root}/{image_meta['file']}",
                       cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bboxes = image_boxes[["xmin", "ymin", "xmax", "ymax"]].values
        bboxes = torch.tensor(bboxes)
        if "labels" in image_boxes.columns:
            labels = torch.tensor(image_boxes["labels"].values,
                                  dtype=torch.int64)
        else:
            labels = torch.tensor(np.ones(bboxes.shape[0]),
                                  dtype=torch.int64)
        return image, bboxes, labels
