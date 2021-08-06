import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split


def get_train_and_val(annotation_files,
                      image_dirs,
                      rand=6189,
                      min_frac=0.18,
                      max_frac=0.22):
    annotation_list = []
    seq_list = []
    col_names = ["frame_id", "object_id", "x", "y", "width", "height", "object_class",
                 "species", "occluded", "noisy_frame"]
    for annot, image_dir in zip(annotation_files, image_dirs):
        if ".csv" in annot:
            annotation_list.append(pd.read_csv(annot, header=None,
                                               names=col_names))
            annotation_list[-1]['csv'] = annot.split("/")[-1]
            images = [f"{image_dir}/{d}" for d in os.listdir(image_dir)]
            img_shape = cv2.imread(images[0]).shape[:2]
            frames = [int(img.split("_")[-1].split(".")[0]) for img in images]
            frame_df = pd.DataFrame({"frame_id": frames, "file": images})
            frame_df["csv"] = f"{image_dir.split('/')[-1]}.csv"
            frame_df["img_height"] = img_shape[0]
            frame_df["img_width"] = img_shape[1]
            seq_list.append(frame_df)
    frame_df = pd.concat(seq_list).reset_index(drop=True)
    annotations = pd.concat(annotation_list).merge(frame_df).reset_index(drop=True)

    while True:
        train_files, val_files = train_test_split(frame_df[['csv']].drop_duplicates(),
                                                  test_size=0.2,
                                                  random_state=rand)
        val_df = frame_df[frame_df['csv'].isin(val_files['csv'])].reset_index(drop=True)
        train_df = frame_df[frame_df['csv'].isin(train_files['csv'])].reset_index(drop=True)
        val_frac = val_df['file'].nunique()/frame_df['file'].nunique()
        if min_frac > val_frac > max_frac:
            break
        rand += 1
    train_annotations = annotations[annotations['csv'].isin(train_files['csv'])].reset_index(drop=True)
    val_annotations = annotations[annotations['csv'].isin(val_files['csv'])].reset_index(drop=True)
    return train_df, val_df, train_annotations, val_annotations