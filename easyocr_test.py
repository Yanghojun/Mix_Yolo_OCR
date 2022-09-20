import cv2
import easyocr

capture = cv2.VideoCapture(2)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
reader = easyocr.Reader(['ko', 'en'], gpu=True, cudnn_benchmark=True, quantize=True)

while 1:
    ret, frame = capture.read()
    print(frame.shape)
    cv2.imshow("VideoFrame", frame)
    result = reader.readtext(frame, batch_size=1, detail=0)
    # print(reader.readtext(frame, batch_size=1, detail=0))
    cv2.waitKey(1)


capture.release()
cv2.destroyAllWindows()