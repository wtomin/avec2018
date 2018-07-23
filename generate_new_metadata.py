import os
import glob
import pandas as pd
old_metadata = '/newdisk/AVEC2018/downloaded_dataset/labels_metadata.csv'
video_dir = '/newdisk/AVEC2018/downloaded_dataset/recordings/recordings_video'
videos = os.listdir(video_dir)
test_videos = sorted([vid.split('.')[0] for vid in videos if vid.startswith('test')])
gender = ['F', 'M', 'M','M', 'F', 'F','M','M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'F',
          'M', 'M', 'F', 'F', 'F', 'M','F', 'M', 'M', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'M',
          'M', 'M', 'M', 'F', 'M', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F',
          'F', 'M', 'M']
test_metadata = pd.DataFrame()
for index, video in enumerate(test_videos):
    test_metadata.loc[index, 'Instance_name'] = video
    test_metadata.loc[index, 'Partition'] = 'test'
    test_metadata.loc[index, 'Gender'] = gender[index]

des = 'test_metadat.csv'
test_metadata.to_csv(des, index=False)
    
