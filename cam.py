from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import pyrealsense2 as rs
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

weights = 'runs\\train\\exp15\\weights\\best.pt'

def rs_cam():
    if torch.cuda.is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(320, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names


    if torch.cuda.is_available():
        model.cuda()
    cudnn.benchmark = True

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipe = rs.pipeline()
    pipe.start(config)
    
    try:
        while True:
            frames = pipe.wait_for_frames()
            depth = frames.get_depth_frame()
            frame = np.array(frames.get_color_frame().get_data())

            frame = torch.from_numpy(frame).to(device)

            frame = cv2.resize(frame, (608, 608))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pred = model(frame, False)
            print(boxes[0])
            if depth is not None:
                result_img = plot_boxes_cv2(frame, boxes[0], savename='any.jpg', class_names=class_names, depth_frame = depth)
            else:
                result_img = plot_boxes_cv2(frame, boxes[0], savename='any.jpg', class_names=class_names)

            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Yolo demo', result_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   


    finally:
        pipe.stop()