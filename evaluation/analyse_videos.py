import pandas as pd
import numpy as np

# This file simply calculates the no slide/slide ratio and jumpiness scores mentioned in the paper.
# In order to do these calculations, the ground truth is needed for a lecture that assigns to every video frame the lecture slide displayed.
# Video frames without a slide displayed are labeled by '-1'.

## calculate ratio between no slide view and slide view:
df1 = pd.read_excel('../ground_truth_files/ground_truth_physics.xlsx')
ground_truth_column = df1['Slidenumber']

no_slide_view = np.count_nonzero(ground_truth_column == -1)
slide_view = np.count_nonzero(ground_truth_column != -1)

print('Number of video frames where no slide is shown: ', no_slide_view)
print('Number of video frames where a slide is shown: ', slide_view)

ratio = no_slide_view/slide_view

print('Ratio of no slide frames to slide frames:', ratio)

## calculate jumpiness:

unique_numbers = len(ground_truth_column.unique())

diffs = np.diff(ground_truth_column)

# Count the number of non-zero differences (i.e., changes)
num_changes = np.count_nonzero(diffs)

# the higher, the 'jumpier' the lecture:
ratio_jumpiness = num_changes/unique_numbers

print('Jumpiness score (Ratio between number of slide changes to unique slide numbers of lecture): ', ratio_jumpiness)
