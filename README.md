[![CircleCI](https://circleci.com/gh/DanielMorton/Petrel/tree/master.svg?style=svg)](https://circleci.com/gh/DanielMorton/Petrel/tree/master)
[![PyPI version](https://badge.fury.io/py/petrel-det.svg)](https://badge.fury.io/py/petrel-det)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/petrel-det.svg)](https://pypi.python.org/pypi/petrel-det/)

# Petrel

PyTorch has a very reliable implementation of [EfficientDet](https://github.com/rwightman/efficientdet-pytorch)
but like all deep learning frameworks it requires a fair amount of work
to prepare the data for use in the model. Petrel streamlines this process
by providing a standard format for loading training and validation data
as well as a single class manage the whole training pipeline. The major
components are described below.

## Input Data Format.

Petrel's Dataset classes take two dataframes as inputs, one consisting
of one row per image, the other consisting of one row per bounding box.

The first table, ```metadata```, is only required to contain one column:
* `file` This should be the absolute path name to the image file. Images
             can be stored in any manner the user wishes.

It is strongly recommended that ```metadata``` rows contain a 
`height` and a `width` column for each image as well. Although
not strictly necessary, it is very likely the preprocessing pipeline
will require this information. Any other image level metadata should be
included in this table, but is not used by this package.

Ensure that the indices for ```metadata``` range from `0` to `numrows - 1`.

A sample table with the minimum recommended columns is below. This data
comes from [NABirds](https://dl.allaboutbirds.org/nabirds); the file
name consists of the species level directory number (encoded elsewhere)
and the random hex string identifying the image file.

|  | file | height | width|
| --- | :---: | :---: | :---: |
| 0 | 0645/0001afd499a14a67b940d419413e23b3.jpg	| 680 | 1024 |
| 1	| 0900/0007181fa7274481ad89591200c61b9d.jpg	| 819 | 1024 |
| 2	| 0988/00071e2081564bd8b5ca6445c2560ee5.jpg	| 768 | 1024 |
| 3	| 0845/00081fce2a744a9fb52b9bb8871a48e2.jpg	| 817 | 1024 |
| 4	| 0698/00085a7befcc4c08a83038477e749101.jpg	| 725 | 1024 |

The second table, ```boxes```, contains the information about the
bounding boxes for each image. Bounding box coordinates are stored
in PASCAL VOC format with each coordinate a separate column. These are
converted into a single tensor by the ```Petrel``` module.
Six columns are required.

* `file` The name of the file containing the bounding box.
* `labels` A numeric label for the class of the object contained in 
           the bounding box. Numbers should range from `1` to
          `num_clases`. Zero is reserved for the empty class.
* `xmin` Upper left horizontal coordinate of the bounding box.
* `ymin` Upper left vertical coordinate of the boundingbox.
* `xmax` Lower right horizontal coordinate of the bounding box.
* `ymax` Lower right vertical coordinate of the bounding box.

A seventh column containing class names, to go with the `label` column,
is useful, but not required. Other columns can be added as the user
requires.

A sample table, this time from the
[Sweden Larch Casebearer](http://lila.science/datasets/forest-damages-larch-casebearer/) 
dataset is below. In this table `damage` is the class name
corresponding to the label.

|  | file | damage | labels	| xmin | ymin | xmax | ymax |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0	| Jallasvag_20190527/Images/B03_0002.JPG | HD | 2 | 205	| 288 | 297 | 380 |
| 1	| Jallasvag_20190527/Images/B03_0002.JPG | HD | 2 | 276	| 186 | 425 | 399 |
| 2 | Kampe_20190527/Images/B04_0053.JPG | LD | 3 | 287 | 817 | 474 | 1017 |
| 3 | Kampe_20190527/Images/B04_0130.JPG | HD | 2 | 537 | 301 | 641	| 411 | 
| 4 | Kampe_20190527/Images/B04_0115.JPG | H  | 1 | 1361 | 567 |1455 | 651 |


## Preparing Training and Validation Data

The classes ```TrainDataset``` and ```ValDataset``` extend the base
```PetrelDataset``` class by adding functionality for separate training
and validation preprocessing. Preprocessing is done by a chained
series of transformations operating on both the image and the bounding
boxes.

Preprocessing for training usually involves iterating through a series
of transformations, some of which are random (i.e. random cropping).
There is usually a requirement that at least one bounding box in the
image be preserved be contained in the final output, otherwise the
preprocessing steps are run again. If this fails after a certain number
of attempts another set of transformations (which may or may not be
the same) is done once in its place. ```TrainDataset``` is designed to
be used with [Albumentations](https://albumentations.ai) chained
together by `Compose`, but could work with another package with a
similar API.

Since validation requires the same images for each epoch
ValDataset` only performs one transformation operation.

Users are expected to provide their own transformation functions as
this is the part of model training that is most dependent on the
dataset in use. A sample transformation that only reshapes images to
a fixed size is provided in `dataset/___init__.py`

A sample `TrainDataset` and `ValDataset` is construction is shown below.

```python
from petrel.dataset import TrainDataset, ValDataset
train_dataset = TrainDataset(
    meta_data=TRAINING_METADATA,
    boxes=TRAINING_BOXES,
    image_root=f"{TRAIN_IMG_ROOT_DIR}",
    transform=TRAIN_TRANSFORM
)

val_dataset = ValDataset(
    meta_data=VAL_METADATA,
    boxes=VAL_BOXES,
    image_root=f"{VAL_IMG_ROOT_DIR}",
    transform=VAL_TRANSFORM,
    train_pipe=True #Set to false if not used for training.
)
```

Setting up and training a model is even simpler.

```python
from petrel.model import load_edet, load_optimizer, load_scheduler, ModelTrainer
model = load_edet(
    "tf_efficientdet_d0",
    image_size=512,
    num_classes=4
)
optimizer = load_optimizer(
    "adamw",
    model,
    learning_rate=2.56e-3
)
scheduler = load_scheduler(
    "exponential",
    optimizer=optimizer,
    gamma=0.94**0.25
)
model_trainer = ModelTrainer(
    model,
    optimizer,
    scheduler,
    base_dir=f"{BASE_MODEL_DIR}"
)
```

## Prediction and Evaluation

The ```predict``` module contains code for conputing predictions for
single batches, which can be size `1`, or whole datasets.
The `predict` function takes an image and an optional dictionary
of bounding boxes and labels and returns a dictionary of bounding
box predictions. When a full batch job is required, `predict_df` or
`val_predict_df` can be used depending on whether ground truth data
is available.

The ```eval``` module computes evaluation metrics. For a range of IOU
match thresholds, defaulting to the COCO standard `0.50:0.05:0.95`,
`eval` computes the PASCAL-VOC mAP for each class and an unweighted
average mAP over all classes. These can then be averaged over the IOU
thresholds to approximate the COCO mAP (COCO mAP averages over 
regular recall values `0.01:0.01:0.99` while PASCAL-VOC mAP is a
continous integral over recall values; the difference will be slight.)

Below is an example of evaluating a model on a dataset.

```python
from petrel.model import load_edet
from petrel.predict import model_eval, val_prediction_df
model = load_edet(
    "tf_efficientdet_d0",
    image_size=512,
    num_classes=4,
    checkpoint_path=f"{BASE_MODEL_DIR}",
    train=False)

pred_df = val_prediction_df(model, val_loader, verbose=20)
eval_df = model_eval(pred_df, categories=categories)
```

The output of `model_eval` for a simple EfficientDet0 on the Larch
Casebearer Dataset is shown below.

| IOU Threshold| Precision | H | HD | LD | other |
| --- | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 70.74 | 54.01 | 70.91 | 86.00 | 72.06 |
| 0.55 | 69.66 | 53.69 | 69.73 | 84.94 | 70.29 |
| 0.60 | 67.72 | 52.65 | 67.83 | 83.20 | 67.19 |
| 0.65 | 64.78 | 51.01 | 64.51 | 80.51 | 63.09 |
| 0.70 | 59.55 | 48.56 | 58.37 | 75.33 | 55.94 |
| 0.75 | 50.14 | 41.72 | 47.21 | 65.74 | 45.87 | 
| 0.80 | 35.41 | 30.73 | 30.42 | 48.64 | 31.86 |
| 0.85 | 17.29 | 15.68 | 12.02 | 25.28 | 16.16 |
| 0.90 |  3.62 |  3.61 |  1.83 |  5.18 |  3.87 |
| 0.95 |  0.08 |  0.06 |  0.03 |  0.11 |  0.11 |

Averaged over the IOU Thresholds the (approximate) COCO mAP results are

| Category | mAP |
| --- | :---: |
| Precision   | 43.90 |
| Healthy     | 35.17 |
| High Damage | 42.29 |
| Low Damage  | 55.49 |
| Other       | 42.64 |