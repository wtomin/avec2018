import tqdm
import glob
import os
import pdb
from tf_pose import common

import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import subprocess
import pandas as pd
video_path = '/newdisk/AVEC2018/downloaded_dataset/recordings/recordings_video'
videos = sorted(glob.glob(os.path.join(video_path, '*.mp4')))
pose_path = '/newdisk/AVEC2018/downloaded_dataset/recordings/video_pose'
if not os.path.exists(pose_path):
    os.mkdir(pose_path)
pdb.set_trace()
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
for vid in tqdm.tqdm(videos):
    pass
    video_name = vid.split('/')[-1].split('.')[0]
    csv_head = ['instance_name', 'frameTime']
    df= pd.DataFrame(columns=csv_head)
    frame_store_path = os.path.join(pose_path, video_name)
    if not os.path.exists(frame_store_path):
        os.mkdir(frame_store_path)
        cmd = ['ffmpeg', '-i', vid, '-filter:v', 'fps=fps=30', os.path.join(frame_store_path, '%05d.jpg')]
        subprocess.call(cmd)
    images = sorted(glob.glob(os.path.join(frame_store_path,'*.jpg')))
    image_num = len(images)
    csv_path = os.path.join(pose_path, video_name+'.csv')
    if not os.path.exists(csv_path):
        for index, img in enumerate(images):
            print('\r','current frame is {}/{}'.format(index, image_num),end="")
            img = common.read_imgfile(img, None, None)
            humans = e.inference(img, resize_to_default=True, upsample_size=4.0)
            df.loc[index,'frameTime'] = index*(1/30.0)
            if len(humans)!=0:
                human = humans[0]
                body_parts = human.body_parts
                for key in body_parts.keys():
                    body_part = body_parts[key]
                    names = [str(body_part.part_idx)+'_x', str(body_part.part_idx)+'_y']
                    items  = [body_part.x, body_part.y]
                    for name, item in zip(names, items):
                        df.loc[index, name] = item
            df.loc[index, 'instance_name'] = video_name
    
        df.to_csv(csv_path,  index=False)