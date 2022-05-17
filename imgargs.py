import os
import argparse
from typing import AsyncGenerator
import matplotlib.pyplot as plt
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

parser = argparse.ArgumentParser(description="--input_path: Insert original file directory. \n--output_path: Insert directory that want to save arg img \n")

parser.add_argument('--input_root', type=str, required=True, help="Insert original file directory")
parser.add_argument('--input_label_root', type=str, required=True, help="Insert original label file directory")
parser.add_argument('--output_root', type=str, required=False, default='./output', help="Insert directory that want to save arg img")
parser.add_argument('--output_label_root', type=str, required=False, default='./output_label', help="Insert directory that want to save arg label")
parser.add_argument('--num', type=int, required= False, default= 2, help= "insert how many times make arg img")

args = parser.parse_args()

input_root = args.input_root
input_label_root = args.input_label_root
output_root = args.output_root
output_label_root = args.output_label_root
make_time = args.num

def make_arg(input_image, output_path, img_path, label_dir, output_label_path):
    """ Augmentation 된 이미지 저장

    Args:
        input_image (numpy): 입력 이미지
        output_path (str): 저장할 이미지 파일 이름
        label_dir (str): 레이블들 저장되어 있는 root dir
    """    

    # 호준 레이블 어그멘테이션 코드
    img_height, img_width = input_image.shape[1], input_image.shape[2]

    print(os.path.splitext(os.path.split(img_path)[-1]))
    label_path = os.path.join(label_dir, os.path.splitext(os.path.split(img_path)[-1])[0] + '.txt')

    if not os.path.exists(label_path):
        print(f"{label_path}인 레이블 파일이 존재하지 않습니다.")
        return

    with open(label_path) as f:
        data = f.readlines()
    
    coor_list = []
    # 여러 bounding box 정보를 담기 위함
    for idx, bbox in enumerate(data):
        cls, cx, cy, norm_width, norm_height = map(float, bbox.split())

        # PIL에 맞게 스케일링
        _x1, _y1, _x2, _y2 = (cx - norm_width/2), (cy - norm_height/2), (cx + norm_width/2), (cy + norm_height/2)
        _x1 *= img_width
        _x2 *= img_width
        _y1 *= img_height
        _y2 *= img_height

        coor_list.append([cls, _x1, _y1, _x2, _y2, norm_width, norm_height])    # norm_width, norm_height는 좌표 복원할 때 쓰임    
                                                                                # norm 붙인 이유는 0 ~ 1 까지 범위로 스케일링 되어있기 때문
    # 리스트 컴프리헨션 이용하여 여러개의 box 있는 경우 다 그림
    bbs = BoundingBoxesOnImage([
    BoundingBox(x1=_x1, y1=_y1, x2=_x2, y2=_y2) for _, _x1, _y1, _x2, _y2, _, _ in coor_list
    ], shape=input_image.squeeze().shape)

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                # iaa.SimplexNoiseAlpha(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                # ])),
                iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    # iaa.FrequencyNoiseAlpha(
                    #     exponent=(-4, 0),
                    #     first=iaa.Multiply((0.5, 1.5), per_channel=True),
                    #     second=iaa.LinearContrast((0.5, 2.0))
                    # )
                    iaa.BlendAlphaFrequencyNoise(
                        exponent=(-4, 0),
                        foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                        background=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
    )

    image_aug, bbs_aug = seq(images=input_image, bounding_boxes=bbs)
    plt.imsave(output_path,image_aug.squeeze())

    img_height, img_width = image_aug.shape[1], image_aug.shape[2]

    annotation = []

    for idx,bbox in enumerate(bbs_aug):
        # cls, norm_width, norm_height = int(coor_list[idx][0]), img_width, img_height
        cls = int(coor_list[idx][0])
        _x1, _y1, _x2, _y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        # augmentation 된 이미지에서 bbox 포인트 0~1로 normalization
        _x1 /= img_width
        _x2 /= img_width
        _y1 /= img_height
        _y2 /= img_height

        # normalization된 bbox 포인트 이용해서 normalization된 width, height 추출
        norm_width = _x2 - _x1      
        norm_height = _y2 - _y1

        cx, cy = (_x1 + norm_width/2), (_y1 + norm_height/2)

        # annotation 리스트에 텍스트 파일로 저장할 정보 저장
        annotation.append(f'{cls} {cx} {cy} {norm_width} {norm_height}\n')

    # 텍스트 파일 작성
    label_save_path = os.path.join(output_label_path, os.path.splitext(os.path.split(output_path)[-1])[0] + '.txt')

    with open(label_save_path, 'w') as f:
        for idx, row in enumerate(annotation):
            # print(f"재현된 {idx} 번째 BBOX: {row}")
            f.write(row)



if __name__ == '__main__':
    current_num = 0
    all_num = 0
    for (root, directories, files) in os.walk(input_root):
        for file in files:
            filename, ext = os.path.splitext(file)
            if ext in IMG_EXTENSIONS:
                all_num += 1

    all_num *= make_time

    if not os.path.exists(input_root):
        print("invalid input path")
        exit()

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    if not os.path.exists(output_label_root):
        os.makedirs(output_label_root)

    for (root, directories, files) in os.walk(input_root):
        middle_path = root.replace(input_root,'')
        if len(middle_path) > 0:
            if (middle_path[0] == '/' or middle_path[0] == '\\'):
                middle_path = middle_path[1:]
        output_path = os.path.join(output_root,middle_path)

        for directory in directories:
            if not os.path.isdir(os.path.join(output_root,middle_path,directory)):
                os.makedirs(os.path.join(output_root,middle_path,directory))
    
        for file in files:
            filename, ext = os.path.splitext(file)
            if ext in IMG_EXTENSIONS:
                img_path = os.path.join(root,file)   
                print(img_path)             
                input_image = np.expand_dims(Image.open(img_path),axis=0)
                if input_image.shape[3] == 4:
                    input_image = input_image[:,:,:,:3]
                for i in range(1,make_time+1):
                    output_image = os.path.join(output_path,filename + str(i) + ext)

                    make_arg(input_image=input_image, 
                            output_path=output_image, 
                            img_path=img_path, 
                            label_dir=input_label_root, 
                            output_label_path=output_label_root)
                    
                    current_num+=1
                    print(f"\r{current_num} / {all_num}, {int(current_num/all_num * 100)}%",end='')
