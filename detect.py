"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
from logging import error
import multiprocessing
import os
import traceback
from ctypes import c_char_p
from numpy.core.shape_base import block
from gtts import gTTS
from playsound import playsound
import argparse
import sys
import time
from pathlib import Path
import easyocr
import cv2
import torch
import torch.backends.cudnn as cudnn
from multiprocessing import Process, Queue, Pipe, Manager
from dictionary import *
import speech_recognition as sr
import json
import threading
import numpy as np

image = np.array([0])      # Just initialize to use this variable in Multiprocess

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from dictionary import read_text
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

direction,thing='',''
count = 0
num = 0
lst =['0' for _ in range(30000)]
reader = easyocr.Reader(['ko'], gpu=True)

def korean_coco_load(path:str)->dict:
    with open(path, encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        
        return json_data

def speech2text(shared_tracking_class, korean_names, names):
        r = sr.Recognizer()
        r.energy_threshold = 800000
        # Reading Microphone as source
        # listening the speech and store in audio_text variable
        
        while(True):
            with sr.Microphone() as source:  
                # r.adjust_for_ambient_noise(source)    # dynamically adjust ambient noise
                print("Waiting for voice input...")
                # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling

                try:
                    audio_text = r.listen(source, phrase_time_limit=3)
                    # using google speech recognition
                    result = r.recognize_google(audio_text, language="ko-KR")
                    print("Text: "+ result)
                    print(korean_names.values())
                    
                    if result in korean_names.values():
                        print("목록에 존재함")
                        shared_tracking_class.value = names[list(korean_names.values()).index(result)] # Put selected class to tracking list
                        tts = gTTS(result + " 입력 되었습니다", lang='ko')
                        tts.save('./listened_voice.mp3')
                        playsound.playsound('./listened_voice.mp3', block=True)
                        os.remove('./listened_voice.mp3')
                        print(shared_tracking_class.value)
                        
                    else:
                        print("목록에 없음")
                        tts = gTTS("다시 말씀해주세요", lang='ko')
                        tts.save('./listened_voice.mp3')
                        playsound.playsound('./listened_voice.mp3', block=True)
                        print("Is this printed out?")
                        os.remove('./listened_voice.mp3')
                        
                except Exception as e:
                    # print(e)
                    # print("Waiting finished...")
                    traceback.print_exc()

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='record',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    # if device == 'cpu':
    #     reader = easyocr.Reader(['ko'], gpu=False)
    # else:
    #     reader = easyocr.Reader(['ko'], gpu=True)

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    save_img = False    # I don't want to save
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    fps_cal_list = []
    
    # Initialize multiprocess
    multiprocessing.set_start_method('spawn')
    manager = Manager()
    shared_tracking_class = manager.Value(c_char_p, "mouse")

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names    
    korean_names = korean_coco_load('./utils/coco_korean.json')
    
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    mike_pr1 = Process(target=speech2text, args=(shared_tracking_class, korean_names, names))
    mike_pr1.daemon = True
    print("Mike input process start")
    mike_pr1.start()

    # def use_easy_ocr(img, q):
    #     results = reader.readtext(img.squeeze().transpose(1,2,0))
    #     q.put(results)
                
    parent_conn, child_conn = Pipe(duplex=True)
    pr1 = Process(target = use_easy_ocr, args = (child_conn, )) # should input ", " into args when just 1 argument
    pr1.daemon = True
    pr1.start()
    # tmp = q.get()
    # print(tmp)

    for path, img, im0s, depth_frame in dataset:        # 
        t1 = time_synchronized()
        parent_conn.send(img)

        # try:        # Try Catch for avoid UnBoundLocalError problem
        #     results = reader.readtext(img)
        #     print(results)
        #     exit()
        #     for (bbox, text, prob) in results:
        #         if check_dic(text):
        #             if text == '꽉자바':
        #                 text = '깍자바'
        #             tipe, summary, title = get_summary(text)
        #             read_text(text)
        # except Exception as e: 
        #     print(e)
        #     exit()
        #     # pass
            
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)




        
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # t2 = time_synchronized()

        
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        global direction,thing
        global count
        global num
        global lst
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            #frame 화면에 출력
            # FPS ="FPS : %0.1f"%int(1/(t2-t1))
            # cv2.putText(im0, FPS, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255), 1,cv2.LINE_AA)
            
            if len(det):    # if something is(are) detected
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxyq).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:# Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        _direction,_depth=plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness, depth_frame=depth_frame)

                        if names[c] == shared_tracking_class.value and conf > 0.75:
                            tts = gTTS(text = korean_names[names[c]] + _direction + "시 방향에"+ _depth[0:3] + "미터 거리에 있습니다", lang='ko', slow=False)
                            tts.save('./temp_voice.mp3')
                            playsound.playsound('./temp_voice.mp3', block=True)
                            os.remove('./temp_voice.mp3')
                        
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            # my_fps = (int)(1/(t2-t1))
            # print(f'{s}Done. ({my_fps}FPS)')
            
            tmp = parent_conn.recv()
            # print(parent_conn.recv())
            t2 = time_synchronized()
            FPS ="FPS : %0.1f"%int(1/(t2-t1))
            cv2.putText(im0, FPS, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255), 1,cv2.LINE_AA)
            fps_cal_list.append(int(1/(t2-t1)))
            

            if len(fps_cal_list) % 30 == 0:
                print(f"Mean FPS: {sum(fps_cal_list) / len(fps_cal_list)}")
                fps_cal_list = []

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new videoABB
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'


                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

def use_easy_ocr(conn):
    while(True):
        try:
            local_img = conn.recv()
            result = reader.readtext(local_img.squeeze().transpose(1,2,0))
            conn.send(result)
            
        except Exception as e:
            print(e)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
