#!/usr/bin/env python
# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import torch
import sys
from ensemble_boxes import *
import glob

import argparse

from utils.datasets import *
from utils.utils import *
import csv
from wheat_eval import *

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)

def write_gt_csv(gt_path):
    print('gt_path',gt_path)
    txt_paths = glob.glob(gt_path + r"*.txt")

    print('len(txt_paths)',len(txt_paths))
    results = []
    for txt in txt_paths:
        boxes =[]
        scores = []
        result = {}
        image_id = txt.split("/")[-1].split(".")[0]

        with open(txt, encoding='utf-8') as f:  # ',encoding='utf-8'
            lines = f.readlines()
            for line in lines:
                splited = line.strip().split(" ")

                print('splited',splited)

                score = float(splited[0])# img_path + txt[:-4].replace(root_path,'')+'.jpg'
                box = [float(splited[1])*1024,float(splited[2])*1024,float(splited[3])*1024,float(splited[4])*1024]
                scores.append(score)
                boxes.append(box)
            print('scores',scores)
            print('boxes',boxes)

            result = {'image_id': image_id, 'PredictionString': format_prediction_string(boxes, scores)}
        results.append(result)
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv('gt_val.csv', index=False)

def evaluate(gt_csv,pred_csv,write_img,iou_thrs=np.arange(0.5, 0.76, 0.05)):
    """Evaluation in COCO protocol.

    Args:
        results (list): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated.
        logger (logging.Logger | str | None): Logger used for printing
            related information during evaluation. Default: None.
        jsonfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        classwise (bool): Whether to evaluating the AP for each class.
        proposal_nums (Sequence[int]): Proposal number used for evaluating
            recalls, such as recall@100, recall@1000.
            Default: (100, 300, 1000).
        iou_thrs (Sequence[float]): IoU threshold used for evaluating
            recalls. If set to a list, the average recall of all IoUs will
            also be computed. Default: 0.5.

    Returns:
        dict[str: float]
    """
    img_root = '/mei/yolov5/input/global-wheat-detection/JPEGImages/'
    gt_dict ={}
    pred_dict ={}
    f_gt = csv.reader(open(gt_csv, 'r'))
    for i in f_gt:
        if 'image' in i[0]:
            continue
        if len(i[1].split(' '))<4:
            continue
        gt_dict[i[0]] = [int(float(item)) for item in i[1].strip().split(' ')]
    f_pred = csv.reader(open(pred_csv, 'r'))
    for i in f_pred:
        if 'image' in i[0]:
            continue
        if len(i[1].split(' '))<4:
            continue
        pred_dict[i[0]] = [float(item) for item in i[1].strip().split(' ')]

    gt_box = []
    gt_boxes = []
    pred_box = []
    preds = []
    imgs_id = []
    for key in gt_dict:
        if key in pred_dict:
            imgs_id.append(key)
            num_gt_box = int(len(gt_dict[key])/5)
            gt_box.append(np.array([[int(float(gt_dict[key][box_id*5+1])-0.5*float(gt_dict[key][box_id*5+3])),int(float(gt_dict[key][box_id*5+2])-0.5*float(gt_dict[key][box_id*5+4])),int(float(gt_dict[key][box_id*5+1])+0.5*float(gt_dict[key][box_id*5+3])),int(float(gt_dict[key][box_id*5+2])+0.5*float(gt_dict[key][box_id*5+4]))] for box_id in range(num_gt_box)]).astype(np.int))
            num_pred_box = int(len(pred_dict[key])/5)
            pred_box.append([np.array([float(pred_dict[key][box_id*5+1]),float(pred_dict[key][box_id*5+2]),float(pred_dict[key][box_id*5+1])+float(pred_dict[key][box_id*5+3]),float(pred_dict[key][box_id*5+2])+float(pred_dict[key][box_id*5+4])]).astype(np.float) for box_id in range(num_pred_box)])
    gt_boxes = gt_box
    preds = np.array(pred_box)
    image_precisions = []

    for img_id, gt, pred in zip(imgs_id,gt_boxes, preds):
        if write_img:
            imagepath = img_root+img_id+'.jpg'
            image = cv2.imread(imagepath)
            for g, p in zip(gt, pred):
                image = cv2.rectangle(image, (int(g[0]),int(g[1])), (int(g[2]),int(g[3])), (255,0,0), 1)
                image = cv2.rectangle(image, (int(p[0]),int(p[1])), (int(p[2]),int(p[3])), (0,0,255), 1)
            write_name = '/mei/yolov5/input/write_image_path/write_gt_pred_image/' +'gt_pred_' + img_id+'.png'
            cv2.imwrite(write_name,image)

        image_precision = calculate_image_precision(gt, pred, thresholds=iou_thrs, form='pascal_voc')
        image_precisions.append(image_precision)

    image_precisions = np.array(image_precisions)
    print("The average precision of the sample image: {0:.4f}".format(image_precisions.mean()))
    eval_results = {}
    return eval_results

def detect(save_img=False):
    weights, imgsz = opt.weights,opt.img_size
    source = '/mei/kaggle_data/val/'
    # Initialize
    device = torch_utils.select_device(opt.device)
    half = False
    # Load model
    model = torch.load(weights, map_location=device)['model'].to(device).eval()
    dataset = LoadImages(source, img_size=1024)
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    all_path=[]
    all_bboxex =[]
    all_score =[]
    for path, img, im0s, vid_cap in dataset:
        print(im0s.shape)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = torch_utils.time_synchronized()
        bboxes_2 = []
        score_2 = []
        if True:
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, 0.4, opt.iou_thres,fast=True, classes=None, agnostic=False)
            t2 = torch_utils.time_synchronized()

            bboxes = []
            score = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    for *xyxy, conf, cls in det:
                        if True:  # Write to file
                            xywh = torch.tensor(xyxy).view(-1).numpy()  # normalized xywh
#                             xywh[2] = xywh[2]-xywh[0]
#                             xywh[3] = xywh[3]-xywh[1]
                            bboxes.append(xywh)
                            score.append(conf)
            bboxes_2.append(bboxes)
            score_2.append(score)
        all_path.append(path)
        all_score.append(score_2)
        all_bboxex.append(bboxes_2)
    return all_path,all_score,all_bboxex


if __name__ == '__main__':
    gt_path ='/mei/yolov5/input/global-wheat-detection/val_txt/'

    gt_csv = '/mei/det_rs/DetectoRS_2/mmdet/datasets/gt_val.csv'
    pred_csv = '/mei/det_rs/DetectoRS_2/mmdet/datasets/pred_val.csv'

    write_img = False
    ans = evaluate(gt_csv,pred_csv,write_img)

    class opt:
        weights = "/mei/yolov5/input/write_image_path/weights/best.pt"
        img_size = 1024
        conf_thres = 0.1
        iou_thres = 0.94
        augment = True
        device = '0'
        classes=None
        agnostic_nms = True
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        res = detect()


# In[5]:


def run_wbf(boxes,scores, image_size=1024, iou_thr=0.33, skip_box_thr=0.34, weights=None):
    #print('type(boxes)',type(boxes))
    boxes_new = []
    for box_every in boxes:
        boxes_every_new = [box/(image_size-1) for box in box_every]
        boxes_new.append(boxes_every_new)
    #print('boxes_new',boxes_new)
    #exit()
    #boxes_new = [box/(image_size-1) for box in boxes]
    boxes =boxes_new
    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    box_every_after = []
    for box_every in boxes:
        boxes_every_after= [box*(image_size-1) for box in box_every]
        box_every_after.append(boxes_every_after)
    #boxes = boxes*(image_size-1)
    boxes = box_every_after
    return boxes, scores, labels

all_path,all_score,all_bboxex = res

results =[]

size = 300
idx =-1
font = cv2.FONT_HERSHEY_SIMPLEX
image = image = cv2.imread(all_path[idx], cv2.IMREAD_COLOR)
# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2


for row in range(len(all_path)):
    image_id = all_path[row].split("/")[-1].split(".")[0]
    boxes = all_bboxex[row]
    scores = all_score[row]
    boxes, scores, labels = run_wbf(boxes,scores)
    boxes = np.array(boxes)
    boxes = (boxes*1024/1024).astype(np.int32).clip(min=0, max=1023)
    try:
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)}
        results.append(result)
        print('imagepath',all_path[row])
        image = cv2.imread(all_path[row])
        for b, s in zip(boxes, scores):
            image = cv2.rectangle(image, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), (255,0,0), 1)
            image = cv2.putText(image, '{:.2}'.format(s), (b[0]+np.random.randint(20),b[1]), font,
                           fontScale, color, thickness, cv2.LINE_AA)
        write_name = '/mei/yolov5/input/write_image_path/write_image/' +'pred_' + image_id+'.png'
        cv2.imwrite(write_name,image)
    except Exception as e:
        result = {'image_id': image_id, 'PredictionString': format_prediction_string(boxes, scores)}

        # print('result',result)

        results.append(result)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)
test_df.head()

size = 300
idx =-1
font = cv2.FONT_HERSHEY_SIMPLEX 
image = image = cv2.imread(all_path[idx], cv2.IMREAD_COLOR)
# fontScale 
fontScale = 1

color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2

index = 0

for b,s in zip(boxes,scores):
    image = cv2.rectangle(image, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), (255,0,0), 1) 
    image = cv2.putText(image, '{:.2}'.format(s), (b[0]+np.random.randint(20),b[1]), font,  
                   fontScale, color, thickness, cv2.LINE_AA)

    write_name = '/mei/yolov5/input/write_image_path/write_image/' + str(index)+'.png'
    cv2.imwrite(write_name,image)
    index +=1



