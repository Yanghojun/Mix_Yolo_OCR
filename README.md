- [Installation](#installation)
  - [windows 11](#windows-11)
    - [Test Code](#test-code)
- [BLEU 기반 성능 테스트](#bleu-기반-성능-테스트)
  - [사용한 영상](#사용한-영상)
  - [실험결과](#실험결과)
- [Training](#training)
  - [Annotation](#annotation)
  - [Data Augmentation](#data-augmentation)

# Installation

## windows 11
``` python
pip install pyrealsense2
```
- 위 명령어만 실행해주면 바로 아래 테스트 코드 실행 가능

### Test Code
```python
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
```
- 결과화면


![](/images/2022-02-14-00-25-22.png)

# BLEU 기반 성능 테스트

[실험 스크립트](/utils/BLEU_Eval.ipynb)

## 사용한 영상

<table>

<tr align="center"> <!-- 한 줄 생성-->
<td> Cetaphil </td>  <!-- 한 줄에 채울 칼럼 한칸 씩 여는 것 -->
<td> 만능크리너 </td>
</tr>

<tr align="center">
<td>
<img width="60%" src="/videos/Cetaphil.gif">
</td>

<td>
<img width="60%" src="/videos/만능크리너.gif">
</td>
</tr>

<tr align="center">
<td> 정관장 </td>
<td> 딱풀</td>
</tr>

<tr align="center">
<td>
<img width="60%" src="/videos/정관장.gif">
</td>
<td>
<img width="60%" src="/videos/딱풀.gif">
<td>
</tr>

</table>

## 실험결과 

|품목 이름|평균 BLEU 스코어|프레임 수|
|---|---|---|
|Cetaphil|0.295|325|
|만능크리너|0.88|443|
|정관장|0.48|434|
|딱풀|0.12|437|

# Training

```python
python train.py --data ./data/data.yaml --epochs 300 --project yolo_agri_experiment --bbox_interval 5 --save-period 5 --batch-size 64 --workers 16
```
- imgsz의 경우 detect.py 에서는 `(640, 480)` 형식으로 imgsz를 입력해줄 수 있지만 training.py 에서는 하나의 값(정수)로만 입력이 가능하다.

## Annotation

- labelimg 프로그램 사용: 아래 명령어로 labelimg를 열어야 클래스 인덱스 꼬이지 않음. 그렇지 않을 시 pork 클래스가 3번으로 인덱싱 되는것(다른 데이터들은 0으로 pork 인덱싱 되어있는 상태)과 같은 문제 발생

    ```bash
    labelimg [image_dir] [predefined_classes.txt file name]
    ```

    - 현재(2022-05-17)의 predefined_classes.txt file

        ```bash
        pork
        shovel
        rake
        homi
        pick
        bigpick
        ```

## Data Augmentation

- imgargs.py

    ```python
    python imgargs.py --input_root [root dir of images] --input_label_root [root dir of labels] --num [num of augmentation per image]
    ```

  - [x] 138 서버에서 3000 * 5 = 15000장 이미지 및 레이블 파일 생성
