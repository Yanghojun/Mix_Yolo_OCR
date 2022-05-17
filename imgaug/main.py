import imgaug.augmenters as iaa
import cv2
import glob

images_path = glob.glob('./*.jpg')

images = []

# Load Dataset
for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)
    # cv2.imshow("dog_1", img)
    # cv2.waitKey(0)

# Image Augmentations

augmentation = iaa.Sequential([
    # Rotation
    # iaa.Rotate((-30, 30)),       # -30 ~ 30 중에 랜덤으로 할당됨
    
    # Flip 
    # iaa.Fliplr(0.5),    # Horizontal
                        # 0.5로 두면 Flip을 할때도 있고 안할때도 있음
    # iaa.Flipud(0.5),    # vertical

    # Affine
    # iaa.Affine(translate_percent={
    #     "x": (-0.2, 0.2), "y": (-0.2, 0.2)},        # x축, y축 방향으로 각각 -0.2 ~ 0.2중 랜덤값으로 밀림
    #     rotate=(-30, 30),
    #     scale=(0.5, 1.5)),

    # Multiply
    # iaa.Multiply((0.2, 1.7)),       # 이미지 밝기 조절 가능

    # Linearcontrast
    # iaa.LinearContrast((0.6, 1.4),)

    # Sometimes와 결합하면 0.5로 값을 줬으므로 절반 정도만 GaussianBlur 처리하는 것임)
    iaa.Sometimes(0.5,

        # GaussianBlur
        iaa.GaussianBlur((0.0, 3.0))
    )

    
    
])

augmented_images = augmentation(images=images)

# Show images
for img in augmented_images:
    cv2.imshow("Image", img)
    cv2.waitKey(0)