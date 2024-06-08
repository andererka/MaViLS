import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


# Specify the column names
column_names = ['Lecture name', 'OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

# Create an empty DataFrame with these column names
df_precision  = pd.DataFrame(columns=column_names)
df_recall  = pd.DataFrame(columns=column_names)
df_F1  = pd.DataFrame(columns=column_names)

def new_row(row_name='None'):
    df_precision.loc[row_name] = [pd.NA] * len(column_names)  # Initialize with NA or a default value
    df_precision.loc[row_name, 'Lecture name'] = row_name
    df_recall.loc[row_name] = [pd.NA] * len(column_names)  # Initialize with NA or a default value
    df_recall.loc[row_name, 'Lecture name'] = row_name
    df_F1.loc[row_name] = [pd.NA] * len(column_names)  # Initialize with NA or a default value
    df_F1.loc[row_name, 'Lecture name'] = row_name
    return row_name

print('deep learning Goodfellow lecture')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_deeplearning.xlsx')
df2 = pd.read_excel('../results/DL_matching_with_all_jump_penality_0comma1.xlsx')
df3 = pd.read_excel('../results/DL_matching_with_ocr_penality_0comma1.xlsx')
df4 = pd.read_excel('../results/DL_matching_with_audioscript_jp_0comma1.xlsx')
df5 = pd.read_excel('../results/DL_sift_matching.xlsx')
df6 = pd.read_excel('../results/DL_videoframe_matching_jp_0comma1.xlsx')

df8 = pd.read_excel('../results/deep_learning_matching_with_all_0comma2.xlsx')
df9 = pd.read_excel('../results/deep_learning_matching_with_all_0comma1.xlsx')
df10 = pd.read_excel('../results/deep_learning_max_matching_all_0comma0.xlsx')
df11 = pd.read_excel('../results/deep_learning_max_matching_all_0comma1.xlsx')
df12 = pd.read_excel('../results/deep_learning_max_matching_all_0comma2.xlsx')
df13 = pd.read_excel('../results/deep_learning_max_matching_all_0comma15.xlsx')
df14 = pd.read_excel('../results/deep_learning_max_matching_all_0comma25.xlsx')

df_mean_0 = pd.read_excel('../results/deep_learning_mean_matching_all_0comma0.xlsx')

df_ocr_00 = pd.read_excel('../results/deep_learning_ocr_0comma0.xlsx')
df_ocr_01 = pd.read_excel('../results/deep_learning_ocr_0comma1.xlsx')
df_ocr_02 = pd.read_excel('../results/deep_learning_ocr_0comma2.xlsx')

df_sift = pd.read_excel('../results/deep_learning_SIFT.xlsx')

df_weighted_sum0 = pd.read_excel('../results/deep_learning_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/deep_learning_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/deep_learning_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/deep_learning_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df_linear1 = pd.read_excel('../results/deep_learning_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/deep_learning_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/deep_learning_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/deep_learning_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/deep_learning_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/deep_learning_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/deep_learning_max_matching_all_0comma1_linearity0.001.xlsx')

row_name = new_row(row_name = 'Deep learning')

result_dfs = [df_ocr_01, df4, df6, df_mean_0, df9, df10, df11, df12, df13, df14, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum',  'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']

ground_truth_labels = ground_truth_column.unique()

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1

print('short range MIT lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_short_range_mit.xlsx')
df2 = pd.read_excel('../results/short_range_MIT_matching_with_all_0comma1.xlsx')
df3 = pd.read_excel('../results/short_range_MIT_matching_with_all_0comma2.xlsx')
df4 = pd.read_excel('../results/short_range_MIT_matching_with_all_0comma0.xlsx')
df5 = pd.read_excel('../results/short_range_MIT_matching_with_all_0comma1_mean.xlsx')
df6 = pd.read_excel('../results/short_range_MIT_matching_with_all_0comma0_mean.xlsx')
df7 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma25.xlsx')
df_ocr = pd.read_excel('../results/short_range_MIT_0comma1_ocr.xlsx')
df_audio = pd.read_excel('../results/short_range_MIT_0comma1_audiomatching.xlsx')
df_image = pd.read_excel('../results/short_range_MIT_0comma1_image_matching.xlsx')

df_weighted_sum0 = pd.read_excel('../results/short_range_MIT_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/short_range_MIT_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/short_range_MIT_0comma1_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/short_range_MIT_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df_sift = pd.read_excel('../results/short_range_MIT_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/short_range_MIT_max_matching_all_0comma1_linearity0.001.xlsx')

row_name = new_row(row_name = 'Short range')

result_dfs = [df_ocr, df_audio, df_image, df6, df5, df4, df2, df3, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']
ground_truth_column = df1['Slidenumber']

ground_truth_labels = ground_truth_column.unique()

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1


print('numerics lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_numerics.xlsx')
df2 = pd.read_excel('../results/numerics_tuebingen_matching_with_all_0comma0_max.xlsx')
df3 = pd.read_excel('../results/numerics_tuebingen_matching_with_all_0comma1_max.xlsx')
df4 = pd.read_excel('../results/numerics_tuebingen_matching_with_all_0comma2_max.xlsx')
df5 = pd.read_excel('../results/numerics_tuebingen_matching_with_all_0comma1_mean.xlsx')
df6 = pd.read_excel('../results/numerics_tuebingen_matching_with_all_0comma0_mean.xlsx')

df9 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma15.xlsx')
df10 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma25.xlsx')

df7 = pd.read_excel('../results/numerics_tuebingen_0comma1_weighted_sum_matching_all_0comma0.xlsx')
df8 = pd.read_excel('../results/numerics_tuebingen_0comma1_weighted_sum_matching_all_0comma1.xlsx')

df11 = pd.read_excel('../results/numerics_tuebingen_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df12 = pd.read_excel('../results/numerics_tuebingen_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df_sift = pd.read_excel('../results/numerics_tuebingen_0comma1_SIFT.xlsx')

df_image = pd.read_excel('../results/numerics_tuebingen_0comma1_image_matching.xlsx')
df_audio = pd.read_excel('../results/numerics_tuebingen_0comma1_audiomatching.xlsx')
df_ocr = pd.read_excel('../results/numerics_tuebingen_0comma1_ocr.xlsx')

df_linear1 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/numerics_tuebingen_max_matching_all_0comma1_linearity1e-08.xlsx')

result_dfs = [df_ocr, df_audio, df_image, df6, df5, df2, df3, df4, df9, df10, df7, df8, df_sift, df11, df12, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]  #, df3, df4]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']
ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()

row_name = new_row(row_name = 'Numerics')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1


print('reinforcement learning lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_reinforcement_learning_silver.xlsx')
df2 = pd.read_excel('../results/reinforcement_matching_with_all_0comma1_max.xlsx')
df3 = pd.read_excel('../results/reinforcement_matching_with_all_0comma1_mean.xlsx')
df4 = pd.read_excel('../results/reinforcement_matching_with_all_0comma0_mean.xlsx')
df5 = pd.read_excel('../results/reinforcement_0comma1_mean_matching_all.xlsx')
df6 = pd.read_excel('../results/reinforcement_0comma1_audiomatching.xlsx')
df7 = pd.read_excel('../results/reinforcement_0comma1_image_matching.xlsx')
df8 = pd.read_excel('../results/reinforcement_0comma1_ocr.xlsx')
df9 = pd.read_excel('../results/reinforcement_matching_with_all_0comma0_max.xlsx')
df10 = pd.read_excel('../results/reinforcement_matching_with_all_0comma2_max.xlsx')
df11 = pd.read_excel('../results/reinforcement_max_matching_all_0comma15.xlsx')
df12 = pd.read_excel('../results/reinforcement_max_matching_all_0comma25.xlsx')

df_weighted_sum0 = pd.read_excel('../results/reinforcement_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1= pd.read_excel('../results/reinforcement_0comma1_weighted_sum_matching_all_0comma1.xlsx')
df_sift = pd.read_excel('../results/reinforcement_SIFT.xlsx')

df_weighted_sum3= pd.read_excel('../results/reinforcement_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4= pd.read_excel('../results/reinforcement_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df_linear1 = pd.read_excel('../results/reinforcement_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/reinforcement_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/reinforcement_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/reinforcement_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/reinforcement_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/reinforcement_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/reinforcement_max_matching_all_0comma1_linearity0.001.xlsx')

result_dfs = [df8, df6, df7, df4, df5, df9, df2, df10, df11, df12, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]  
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']

ground_truth_labels = ground_truth_column.unique()

row_name = new_row(row_name = 'Reinforcement')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1

print('computer vision lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_computer_vision_2_2.xlsx')
df2 = pd.read_excel('../results/computer_vision_2_2_matching_with_all_0comma1.xlsx')
df3 = pd.read_excel('../results/computer_vision_2_2_matching_with_all_0comma0.xlsx')
df4 = pd.read_excel('../results/computer_vision_max_matching_all_0comma2.xlsx')
df5 = pd.read_excel('../results/computer_vision_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/computer_vision_max_matching_all_0comma0.xlsx')
df7 = pd.read_excel('../results/computer_vision_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/computer_vision_max_matching_all_0comma25.xlsx')

df_ocr= pd.read_excel('../results/computer_vision_ocr_0comma1.xlsx')
df_audio= pd.read_excel('../results/computer_vision_audiomatching_0comma1.xlsx')
df_image= pd.read_excel('../results/computer_vision_image_matching_0comma1.xlsx')

df_weighted_sum0= pd.read_excel('../results/computer_vision_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1= pd.read_excel('../results/computer_vision_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3= pd.read_excel('../results/computer_vision_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4= pd.read_excel('../results/computer_vision_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift= pd.read_excel('../results/computer_vision_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/computer_vision_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/computer_vision_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/computer_vision_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/computer_vision_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/computer_vision_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/computer_vision_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/computer_vision_max_matching_all_0comma1_linearity0.001.xlsx')

result_dfs = [df_ocr, df_audio, df_image, df3, df2, df6, df5, df4, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]  
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']
ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
print(ground_truth_labels)

row_name = new_row(row_name = 'Computer Vision')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1


print('cities and climate lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_climate_and_cities.xlsx')
df2 = pd.read_excel('../results/cities_and_climate_matching_with_all_0comma0_max.xlsx')
df3 = pd.read_excel('../results/cities_and_climate_matching_with_all_0comma1_max.xlsx')
df4 = pd.read_excel('../results/cities_and_climate_matching_with_all_0comma1_mean.xlsx')
df4_1 = pd.read_excel('../results/cities_and_climate_matching_with_all_0comma0_mean.xlsx')
df5 = pd.read_excel('../results/cities_and_climate_matching_with_all_0comma2_max.xlsx')
df_ocr = pd.read_excel('../results/cities_and_climate_ocr_0comma1.xlsx')
df_audio = pd.read_excel('../results/cities_and_climate_audiomatching_0comma1.xlsx')
df_image = pd.read_excel('../results/cities_and_climate_image_matching_0comma1.xlsx')

df_weighted_sum0 = pd.read_excel('../results/cities_and_climate_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/cities_and_climate_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/cities_and_climate_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/cities_and_climate_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df_linear1 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma1_linearity0.001.xlsx')

df_sift = pd.read_excel('../results/cities_and_climate_SIFT.xlsx')

df6 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma15.xlsx')
df7 = pd.read_excel('../results/cities_and_climate_max_matching_all_0comma25.xlsx')

df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

result_dfs = [df_ocr, df_audio, df_image, df4_1, df4, df2, df3, df5, df6, df7, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()

row_name = new_row('Climate & Cities')


for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1



print('cities and decarbonization lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_cities_and_decarbonization.xlsx')
df2 = pd.read_excel('../results/cities_and_decarbonization_matching_with_all_0comma0_max.xlsx')
df3 = pd.read_excel('../results/cities_and_decarbonization_matching_with_all_0comma0_mean.xlsx')
df4 = pd.read_excel('../results/cities_and_decarbonization_matching_with_all_0comma1_max.xlsx')
df5 = pd.read_excel('../results/cities_and_decarbonization_matching_with_all_0comma1_mean.xlsx')
df6 = pd.read_excel('../results/cities_and_decarbonization_matching_with_all_0comma2_max.xlsx')
df_ocr = pd.read_excel('../results/cities_and_decarbonization_ocr_0comma1.xlsx')
df_audio = pd.read_excel('../results/cities_and_decarbonization_audiomatching_0comma1.xlsx')
df_image = pd.read_excel('../results/cities_and_decarbonization_image_matching_0comma1.xlsx')
df_sift = pd.read_excel('../results/cities_and_decarbonization_SIFT.xlsx')

df_weighted_sum0 = pd.read_excel('../results/cities_and_decarbonization_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/cities_and_decarbonization_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/cities_and_decarbonization_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/cities_and_decarbonization_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df_linear1 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma1_linearity0.001.xlsx')

df7 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/cities_and_decarbonization_max_matching_all_0comma25.xlsx')


result_dfs = [df_ocr, df_audio, df_image, df3, df5, df2, df4, df6, df7, df8, df_weighted_sum0, df_weighted_sum0, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum','All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']
ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()

row_name = new_row('Decarbonization')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1


print('cryptocurrency lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_cryptocurrency_MIT.xlsx')
df2 = pd.read_excel('../results/cryptocurrency_matching_with_all_0comma0_max.xlsx')
df3 = pd.read_excel('../results/cryptocurrency_matching_with_all_0comma0_mean.xlsx')
df4 = pd.read_excel('../results/cryptocurrency_matching_with_all_0comma1_max.xlsx')
df5 = pd.read_excel('../results/cryptocurrency_matching_with_all_0comma1_mean.xlsx')
df6 = pd.read_excel('../results/cryptocurrency_matching_with_all_0comma2_max.xlsx')
df_ocr = pd.read_excel('../results/cryptocurrency_ocr_0comma1.xlsx')
df_audio = pd.read_excel('../results/cryptocurrency_audiomatching_0comma1.xlsx')
df_image = pd.read_excel('../results/cryptocurrency_image_matching_0comma1.xlsx')

df_weighted_sum0 = pd.read_excel('../results/cryptocurrency_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/cryptocurrency_weighted_sum_matching_all_0comma1.xlsx')
df_weighted_sum3 = pd.read_excel('../results/cryptocurrency_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/cryptocurrency_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df7 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma25.xlsx')

df_sift = pd.read_excel('../results/cryptocurrency_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/cryptocurrency_max_matching_all_0comma1_linearity0.001.xlsx')

result_dfs = [df_ocr, df_audio, df_image, df3, df5, df2, df4, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 'All 0.0 weighted sum',
                'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()

row_name = new_row('Cryptocurrency')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1


print('solar resource lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_solar_resource.xlsx')
df2 = pd.read_excel('../results/solar_resource_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/solar_resource_mean_matching_all_0comma1.xlsx')

df4 = pd.read_excel('../results/solar_resource_image_matching_0comma1.xlsx')
df5 = pd.read_excel('../results/solar_resource_audiomatching_0comma1.xlsx')
df6 = pd.read_excel('../results/solar_resource_ocr_0comma1.xlsx')

df7 = pd.read_excel('../results/solar_resource_max_matching_all_0comma0.xlsx')
df8 = pd.read_excel('../results/solar_resource_max_matching_all_0comma1.xlsx')
df9 = pd.read_excel('../results/solar_resource_max_matching_all_0comma2.xlsx')

df12 = pd.read_excel('../results/solar_resource_max_matching_all_0comma15.xlsx')
df13 = pd.read_excel('../results/solar_resource_max_matching_all_0comma25.xlsx')


df10 = pd.read_excel('../results/solar_resource_weighted_sum_matching_all_0comma0.xlsx')
df11 = pd.read_excel('../results/solar_resource_weighted_sum_matching_all_0comma1.xlsx')

df_linear1 = pd.read_excel('../results/solar_resource_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/solar_resource_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/solar_resource_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/solar_resource_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/solar_resource_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/solar_resource_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/solar_resource_max_matching_all_0comma1_linearity0.001.xlsx')

df_weighted_sum3 = pd.read_excel('../results/solar_resource_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/solar_resource_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')


df_sift = pd.read_excel('../results/solar_resource_SIFT.xlsx')


result_dfs = [df4, df5, df6, df2, df3, df7, df8, df9, df12, df13, df10, df11, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']
ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()

row_name = new_row('Solar resource')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1


print('psychology lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_psychology_MIT.xlsx')

df2 = pd.read_excel('../results/psychology_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/psychology_mean_matching_all_0comma1.xlsx')

df4 = pd.read_excel('../results/psychology_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/psychology_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/psychology_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/psychology_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/psychology_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/psychology_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/psychology_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/psychology_ocr_0comma1.xlsx')

df_weighted_sum0 = pd.read_excel('../results/psychology_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/psychology_weighted_sum_matching_all_0comma1.xlsx')
df_weighted_sum3 = pd.read_excel('../results/psychology_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/psychology_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/psychology_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/psychology_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/psychology_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/psychology_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/psychology_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/psychology_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/psychology_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/psychology_max_matching_all_0comma1_linearity0.001.xlsx')

result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']
ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()

row_name = new_row('Psychology')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1



print('creating breakthrough products lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_creating_breakthrough_products_MIT.xlsx')
df2 = pd.read_excel('../results/creating_breakthrough_products_audiomatching_0comma0.xlsx')
df3 = pd.read_excel('../results/creating_breakthrough_products_audiomatching_0comma1.xlsx')
df4 = pd.read_excel('../results/creating_breakthrough_products_image_matching_0comma0.xlsx')
df5 = pd.read_excel('../results/creating_breakthrough_products_image_matching_0comma1.xlsx')
df6 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma0.xlsx')
df7 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma1.xlsx')
df7_1 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma2.xlsx')

df10 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma15.xlsx')
df11 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma25.xlsx')

df8 = pd.read_excel('../results/creating_breakthrough_products_mean_matching_all_0comma0.xlsx')
df9 = pd.read_excel('../results/creating_breakthrough_products_mean_matching_all_0comma1.xlsx')

df_ocr0 = pd.read_excel('../results/creating_breakthrough_products_ocr_0comma0.xlsx')
df_ocr1 = pd.read_excel('../results/creating_breakthrough_products_ocr_0comma1.xlsx')

df_sift = pd.read_excel('../results/creating_breakthrough_products_SIFT.xlsx')
df_weighted_sum0 = pd.read_excel('../results/creating_breakthrough_products_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/creating_breakthrough_products_weighted_sum_matching_all_0comma1.xlsx')
df_weighted_sum3 = pd.read_excel('../results/creating_breakthrough_products_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/creating_breakthrough_products_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df_linear1 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/creating_breakthrough_products_max_matching_all_0comma1_linearity0.001.xlsx')

result_dfs = [df_ocr1,  df3, df5, df8, df9, df6, df7, df7_1, df10, df11, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']
ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()

row_name = new_row('Productdesign')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1


print('image processing lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_image_processing.xlsx')
df2 = pd.read_excel('../results/image_processing_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/image_processing_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/image_processing_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/image_processing_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/image_processing_max_matching_all_0comma2.xlsx')


df11 = pd.read_excel('../results/image_processing_max_matching_all_0comma15.xlsx')
df12 = pd.read_excel('../results/image_processing_max_matching_all_0comma25.xlsx')

df9 = pd.read_excel('../results/image_processing_weighted_sum_matching_all_0comma0.xlsx')
df10 = pd.read_excel('../results/image_processing_weighted_sum_matching_all_0comma1.xlsx')

df13 = pd.read_excel('../results/image_processing_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df14 = pd.read_excel('../results/image_processing_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')

df_ocr = pd.read_excel('../results/image_processing_ocr_0comma1.xlsx')
df_audio = pd.read_excel('../results/image_processing_audiomatching_0comma1.xlsx')
df_image = pd.read_excel('../results/image_processing_image_matching_0comma1.xlsx')

df_sift = pd.read_excel('../results/image_processing_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/image_processing_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/image_processing_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/image_processing_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/image_processing_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/image_processing_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/image_processing_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/image_processing_max_matching_all_0comma1_linearity0.001.xlsx')

result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6,  df11, df12, df9, df10, df_sift, df13, df14, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']
ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()

row_name = new_row('Image processing')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1


print('sensory systems lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_sensory_system.xlsx')
df2 = pd.read_excel('../results/sensory_systems_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/sensory_systems_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/sensory_systems_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/sensory_systems_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/sensory_systems_ocr_0comma1.xlsx')


df_weighted_sum0 = pd.read_excel('../results/sensory_systems_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/sensory_systems_weighted_sum_matching_all_0comma1.xlsx')
df_weighted_sum3 = pd.read_excel('../results/sensory_systems_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/sensory_systems_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/sensory_systems_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/sensory_systems_max_matching_all_0comma1_linearity0.001.xlsx')

result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max', 
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
row_name = new_row('Sensory systems')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1
    

df_precision.to_excel('results_precision.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1.xlsx', index=False, engine='openpyxl')



print('ML for health lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_ML_for_health_MIT.xlsx')
df2 = pd.read_excel('../results/ML_for_health_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/ML_for_health_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/ML_for_health_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/ML_for_health_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/ML_for_health_ocr_0comma1.xlsx')


df_weighted_sum0 = pd.read_excel('../results/ML_for_health_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/ML_for_health_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/ML_for_health_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/ML_for_health_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/ML_for_health_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/ML_for_health_max_matching_all_0comma1_linearity0.001.xlsx')

result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
row_name = new_row('ML for health')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1
    

df_precision.to_excel('results_precision.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1.xlsx', index=False, engine='openpyxl')




print('Climate and policy lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_climate_science_policy_MIT2.xlsx')
df2 = pd.read_excel('../results/climate_and_policies_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/climate_and_policies_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/climate_and_policies_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/climate_and_policies_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/climate_and_policies_ocr_0comma1.xlsx')


df_weighted_sum0 = pd.read_excel('../results/climate_and_policies_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/climate_and_policies_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/climate_and_policies_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/climate_and_policies_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/climate_and_policies_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/climate_and_policies_max_matching_all_0comma1_linearity0.001.xlsx')


result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
row_name = new_row('Climate policies')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1
    

df_precision.to_excel('results_precision.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1.xlsx', index=False, engine='openpyxl')


print('theory of computation lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_theory_of_computation_MIT.xlsx')
df2 = pd.read_excel('../results/theory_of_computation_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/theory_of_computation_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/theory_of_computation_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/theory_of_computation_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/theory_of_computation_ocr_0comma1.xlsx')


df_weighted_sum0 = pd.read_excel('../results/theory_of_computation_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/theory_of_computation_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/theory_of_computation_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/theory_of_computation_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/theory_of_computation_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/theory_of_computation_max_matching_all_0comma1_linearity0.001.xlsx')


result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
row_name = new_row('Theory of Computation')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1
    

df_precision.to_excel('results_precision.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1.xlsx', index=False, engine='openpyxl')


print('Physics lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_physics.xlsx')
df2 = pd.read_excel('../results/physics_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/physics_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/physics_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/physics_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/physics_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/physics_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/physics_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/physics_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/physics_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/physics_ocr_0comma1.xlsx')


df_weighted_sum0 = pd.read_excel('../results/physics_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/physics_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/physics_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/physics_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/physics_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/physics_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/physics_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/physics_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/physics_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/physics_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/physics_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/physics_max_matching_all_0comma1_linearity0.001.xlsx')


result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
row_name = new_row('Physics')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1
    

df_precision.to_excel('results_precision.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1.xlsx', index=False, engine='openpyxl')


print('Phonetics lecture:\n')
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_phonetics.xlsx')
df2 = pd.read_excel('../results/phonetics_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/phonetics_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/phonetics_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/phonetics_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/phonetics_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/phonetics_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/phonetics_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/phonetics_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/phonetics_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/phonetics_ocr_0comma1.xlsx')


df_weighted_sum0 = pd.read_excel('../results/phonetics_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/phonetics_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/phonetics_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/phonetics_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/phonetics_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/phonetics_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/phonetics_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/phonetics_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/phonetics_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/phonetics_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/phonetics_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/phonetics_max_matching_all_0comma1_linearity0.001.xlsx')


result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
row_name = new_row('Phonetics')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1
    

df_precision.to_excel('results_precision.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1.xlsx', index=False, engine='openpyxl')


print('Team dynamics lecture:\n')  
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_team_dynamics_game_design_MIT.xlsx')
df2 = pd.read_excel('../results/team_dynamics_game_design_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/team_dynamics_game_design_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/team_dynamics_game_design_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/team_dynamics_game_design_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/team_dynamics_game_design_ocr_0comma1.xlsx')


df_weighted_sum0 = pd.read_excel('../results/team_dynamics_game_design_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/team_dynamics_game_design_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/team_dynamics_game_design_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/team_dynamics_game_design_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/team_dynamics_game_design_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/team_dynamics_game_design_max_matching_all_0comma1_linearity0.001.xlsx')


result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
row_name = new_row('Team dynamics')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1
    

df_precision.to_excel('results_precision.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1.xlsx', index=False, engine='openpyxl')


print('Cognitive robotics lecture:\n')  
# Load the Excel files
df1 = pd.read_excel('../ground_truth_files/ground_truth_cognitive_robotics_MIT_control.xlsx')
df2 = pd.read_excel('../results/cognitive_robotics_mean_matching_all_0comma0.xlsx')
df3 = pd.read_excel('../results/cognitive_robotics_mean_matching_all_0comma1.xlsx')
df4 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma0.xlsx')
df5 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma1.xlsx')
df6 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma2.xlsx')

df7 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma15.xlsx')
df8 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma25.xlsx')

df_image = pd.read_excel('../results/cognitive_robotics_image_matching_0comma1.xlsx')
df_audio = pd.read_excel('../results/cognitive_robotics_audiomatching_0comma1.xlsx')
df_ocr = pd.read_excel('../results/cognitive_robotics_ocr_0comma1.xlsx')


df_weighted_sum0 = pd.read_excel('../results/cognitive_robotics_weighted_sum_matching_all_0comma0.xlsx')
df_weighted_sum1 = pd.read_excel('../results/cognitive_robotics_weighted_sum_matching_all_0comma1.xlsx')

df_weighted_sum3 = pd.read_excel('../results/cognitive_robotics_weighted_sum_matching_all_0comma1_50iterations.xlsx')
df_weighted_sum4 = pd.read_excel('../results/cognitive_robotics_weighted_sum_matching_all_0comma1_50iterations_with_adam.xlsx')
df_sift = pd.read_excel('../results/cognitive_robotics_SIFT.xlsx')

df_linear1 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma1_linearity0.xlsx')
df_linear2 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma1_linearity0.0001.xlsx')
df_linear3 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma1_linearity1e-05.xlsx')
df_linear4 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma1_linearity1e-06.xlsx')
df_linear5 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma1_linearity1e-07.xlsx')
df_linear6 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma1_linearity1e-08.xlsx')
df_linear7 = pd.read_excel('../results/cognitive_robotics_max_matching_all_0comma1_linearity0.001.xlsx')


result_dfs = [df_ocr, df_audio, df_image, df2, df3, df4, df5, df6, df7, df8, df_weighted_sum0, df_weighted_sum1, df_sift, df_weighted_sum3, df_weighted_sum4, df_linear1, df_linear2,
              df_linear3, df_linear4, df_linear5, df_linear6, df_linear7]
df_names = ['OCR 0.1', 'Audio 0.1', 'Images 0.1', 'All 0.0 mean', 'All 0.1 mean', 
                'All 0.0 max', 'All 0.1 max', 'All 0.2 max', 'All 0.15 max', 'All 0.25 max',
                'All 0.0 weighted sum', 'All 0.1 weighted sum', 'SIFT', 'weighted sum 50', 'weighted sum adams', '0', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001']

ground_truth_column = df1['Slidenumber']
ground_truth_labels = ground_truth_column.unique()
row_name = new_row('Cognitive robotics')

for df, name in zip(result_dfs, df_names):
    result_column = df['Value']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    print(f'Algorithm: {name}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}\n\n')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1
    

df_precision.to_excel('results_precision.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1.xlsx', index=False, engine='openpyxl')