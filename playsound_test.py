import yaml
import playsound
from glob import glob
# with open('speak_info.yaml') as f:
#     tp = yaml.load(f, Loader=yaml.FullLoader)

# print(type(tp['pork']))

# print(len(glob('./mp3_dir/*.mp4')))
playsound.playsound('mp3_dir/pork900.mp4')