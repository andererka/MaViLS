import pandas as pd
import numpy as np


## calculate ratio between no slide view and slide view:
df1 = pd.read_excel('../ground_truth_files/ground_truth_physics.xlsx')
ground_truth_column = df1['Slidenumber']

no_slide_view = np.count_nonzero(ground_truth_column == -1)
slide_view = np.count_nonzero(ground_truth_column != -1)

print(no_slide_view)
print(slide_view)

ratio = no_slide_view/slide_view

print('ratio:', ratio)

## calculate jumpiness:

unique_numbers = len(ground_truth_column.unique())

diffs = np.diff(ground_truth_column)

# Count the number of non-zero differences (i.e., changes)
num_changes = np.count_nonzero(diffs)

print(num_changes)
print(unique_numbers)
# the higher, the 'jumpier' the lecture:
ratio_jumpiness = num_changes/unique_numbers

print('ratio jumpiness: ', ratio_jumpiness)
