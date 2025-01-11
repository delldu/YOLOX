"""YOLOX Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2024年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

import os
from tqdm import tqdm
import torch
import torchvision

from .yolox import YOLOX
from .visualize import draw_boxes

import todos
import pdb

def postprocess(prediction, num_classes=80, conf_thre=0.25, nms_thre=0.45):
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next
        if not image_pred.size(0):
            continue
        conf_mask = (image_pred[:, 4] >= conf_thre).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_score * class_score, class_id)
        # detections = torch.cat((image_pred[:, :5], image_pred[:, 5:6], image_pred[:, 6:7]), 1)
        detections = image_pred[conf_mask]
        if not detections.size(0):
            continue

        # torchvision.ops.nms(boxes: Tensor, scores: Tensor, iou_threshold: float) → Tensor[source]
        #    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).
        #    NMS iteratively removes lower scoring boxes which have an IoU greater than iou_threshold
        #        with another (higher scoring) box.

        #    If multiple boxes have the exact same score and satisfy the IoU criterion with respect
        #        to a reference box, the selected box is not guaranteed to be the same between CPU and GPU.
        #        This is similar to the behavior of argsort in PyTorch when repeated values are present.

        #    Parameters:
        #            boxes (Tensor[N, 4])) – boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2)
        #                format with 0 <= x1 < x2 and 0 <= y1 < y2.
        #            scores (Tensor[N]) – scores for each one of the boxes
        #            iou_threshold (float) – discards all overlapping boxes with IoU > iou_threshold
        #    Returns:
        #        int64 tensor with the indices of the elements that have been kept by NMS,
        #            sorted in decreasing order of scores
        #    Return type:
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],  # (x1, y1, x2, y2)
            detections[:, 4],  # scores
            nms_thre,
        )

        output[i] = detections[nms_out_index]
    #  output[0].size() -- [3, 7]

    return output


def get_yolox_model():
    model = YOLOX()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if "cpu" in str(device.type):
        model.float()

    print(f"Running on {device} ...")

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;
    # # torch::jit::setTensorExprFuserEnabled(false);
    # todos.data.mkdir("output")
    # if not os.path.exists("output/yolox.torch"):
    #     model.save("output/yolox.torch")

    return model, device


def yolox_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_yolox_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        input_tensor = todos.data.load_tensor(filename)
        with torch.no_grad():
            output_tensor = model(input_tensor.to(device))

        # tensor [output_tensor] size: [1, 8400, 6], min: -51.966682, max: 866.434326, mean: 226.630188
        output_tensor = postprocess(output_tensor)
        # tensor [item] size: [4, 6], min: 0.378367, max: 688.59552, mean: 152.347595
        input_tensor = draw_boxes(input_tensor, output_tensor[0])

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        todos.data.save_tensor([input_tensor], output_file)

    todos.model.reset_device()
