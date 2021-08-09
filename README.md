[![CircleCI](https://circleci.com/gh/DanielMorton/Petrel/tree/master.svg?style=svg)](https://circleci.com/gh/DanielMorton/Petrel/tree/master)

# Petrel
Code to streamline Pytorch EfficientDet applications

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

A sample table with the minimum recommended columns is below.

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

A sample table is below.

|  | file | labels | xmin | ymin | xmax | ymax |
| --- | :---: | :---: | :---: | :---: | :---: |:---: |
| 0	| 0645/0001afd499a14a67b940d419413e23b3.jpg	| 4	| 307 | 179	| 799 | 403 | 
| 1	| 0900/0007181fa7274481ad89591200c61b9d.jpg	| 22 | 47 |194 | 866 | 767 | 
| 2	| 0988/00071e2081564bd8b5ca6445c2560ee5.jpg	| 22 | 260 | 146 | 838 | 662 | 
| 3	| 0845/00081fce2a744a9fb52b9bb8871a48e2.jpg	| 22 | 259 | 143 | 915 | 682 | 
| 4	| 0698/00085a7befcc4c08a83038477e749101.jpg	| 1	| 217 | 174	| 898 | 401|


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
together by ```Compose```, but could work with another package with a
similar API.

Since validation requires the same images for each epoch
```ValDataset``` only performs one transformation operation.

Users are expected to provide their own transformation functions as
this is the part of model training that is most dependent on the
dataset in use. A sample transformation that only reshapes images to
a fixed six is provided in ```dataset/___init__.py```.



