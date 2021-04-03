# count statistics of human bbox using detectron2
import os
import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Edit here
PATH_TO_CSV = '/home/ubuntu/data/AVA/annotations/person_box_67091280_iou90/ava_train_predicted_boxes.csv'


chunk = pd.read_csv(PATH_TO_CSV, chunksize=1033086, header=None)
pd_df = pd.concat(chunk)

# Extract rows with common bboxes only
uniq_df = pd_df.drop_duplicates(subset=[0, 1, 2, 3, 4, 5]).copy()
uniq_df = uniq_df.set_axis(['video_id', 'frame_id', 'x1', 'y1', 'x2', 'y2', 'label', 'conf' ], axis=1)
uniq_df['w'] = uniq_df['x2'] - uniq_df['x1']
uniq_df['h'] = uniq_df['y2'] - uniq_df['y1']
uniq_df['area'] = uniq_df['w'] * uniq_df['h']

# Extract the number of bboxes per frame
grouped = uniq_df.groupby(['video_id', 'frame_id'])
grouped_size = grouped.size()

# Draw histograms of (1) # of bboxes,  (2) bbox size
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(bottom=0.2, left=0.1, wspace=0.4)
grouped_size.plot(ax=ax[0], kind='hist', bins=70, range=(0,70))
ax[0].set_xlabel('# of bboxes')
ax[0].set_ylabel('# of frames')
uniq_df['area'].plot(ax=ax[1], kind='hist', bins=100, range=(0, 1), color='orange')
ax[1].set_xlabel('area of bboxes')
ax[1].set_ylabel('count')

# Save to a file
plt.savefig('AVA.png')


    
