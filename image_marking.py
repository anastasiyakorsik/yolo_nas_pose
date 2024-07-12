import os
import sys
import time
import torch
import pathlib

import cv2

from tqdm import tqdm
from typing import Type
from progress.bar import IncrementalBar

from input_processing import InputProcessor
from output_processing import OutputProcessor

from super_gradients.training import models

def add_text_to_image(image_path, text, point):
    image = cv2.imread(image_path)
    x, y = point
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2
    font_color = (0, 0, 0)
    thickness = 1
    cv2.putText(image, text, (x + 10, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.imwrite(image_path, image)


def image_marking(model, img_file):
    """
    Mark model prediction on image, showing links, bboxs coordinates and confidence

    Arguments:
        model: Yolo-NAS Pose model
        img_file (str): Name of image file

    Returns:
        Marked image
    """
    output_file = pathlib.Path(img_file).stem + "-detections" + pathlib.Path(img_file).suffix

    print("Inferencing image")
    model_predictions = model.predict(img_file, conf=0.5)
    model.predict(img_file, conf=0.5).save(output_file)
    print(f"Inferenced image saved as {output_file}")

    prediction = model_predictions.prediction # One prediction per image - Here we work with 1 image, so we get the first.

    bboxes = prediction.bboxes_xyxy # [Num Instances, 4] List of predicted bounding boxes for each object 
    poses  = prediction.poses       # [Num Instances, Num Joints, 3] list of predicted joints for each detected object (x,y, confidence)
    scores = prediction.scores      # [Num Instances] - Confidence value for each predicted instance

    for bbox in bboxes:
        xy1 = (int(bbox[0]), int(bbox[1]))
        xy1_str = str(xy1)
        add_text_to_image(output_file, xy1_str, xy1)

        xy2 = (int(bbox[2]), int(bbox[3]))
        xy2_str = str(xy2)
        add_text_to_image(output_file, xy2_str, xy2)

    for pose in poses:
        for point in pose:
            kp = (int(point[0]), int(point[1]))
            conf = round(float(point[2]), 2)
            kp_str = f"({int(point[0])}, {int(point[1])}); conf:{conf}"
            # kp_str = str((int(point[0]), int(point[1]), point[2]))
            add_text_to_image(output_file, kp_str, kp)


def main():
    try:
        if len(sys.argv) < 2:
            raise ValueError('ERROR - Please provide path to image file')
        
        img_file = sys.argv[1]

        print("INFO - Getting model:")
        model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        image_marking(model, img_file)
            
    except Exception as err:
        print(f"ERROR - Exception occured in main() {err=}, {type(err)=}")
        raise


if __name__ == "__main__":
    main()