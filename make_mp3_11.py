from gtts import gTTS
import yaml
import os
import numpy as np

with open('speak_info.yaml') as f:
    korean_agri_name = yaml.load(f, Loader=yaml.FullLoader)

# def cr(eng, direction, distance):
#     tts = gTTS(text=eng + str(direction) +"시 방향"+ str(distance) + "미터 거리에 있어요", lang='ko')
#     tts.save(os.path.join('./mp3_dir', eng + str(direction) + str(distance)[0] + str(distance)[2] + ".mp4"))

for eng in korean_agri_name.keys():
    for direction in [11]:
        for distance in list(np.arange(0, 10, 0.1)):
            tts = gTTS(text=eng + str(direction) +"시 방향"+ str(distance)[:3] + "미터 거리에 있어요", lang='ko')
            tts.save(os.path.join('./mp3_dir', eng + str(direction) + str(distance)[0] + str(distance)[2] + ".mp4"))

# for name in names:

# tts = gTTS(text=korean_agri_name[names[c]] + direction +"시 방향"+ str(distance)[:3] + "미터 거리에 있어요", lang='ko')