import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


# Specify the column names
column_names = ['Lecture name', 'Consistency']

def new_row(row_name='None'):
    df_precision.loc[row_name] = [pd.NA] * len(column_names)  # Initialize with NA or a default value
    df_precision.loc[row_name, 'Lecture name'] = row_name
    df_recall.loc[row_name] = [pd.NA] * len(column_names)  # Initialize with NA or a default value
    df_recall.loc[row_name, 'Lecture name'] = row_name
    df_F1.loc[row_name] = [pd.NA] * len(column_names)  # Initialize with NA or a default value
    df_F1.loc[row_name, 'Lecture name'] = row_name
    return row_name

# Create an empty DataFrame with these column names
df_precision  = pd.DataFrame(columns=column_names)
df_recall  = pd.DataFrame(columns=column_names)
df_F1  = pd.DataFrame(columns=column_names)

df_physics2 = pd.read_excel('../ground_truth_files/physics_ground_truth_labeled_control.xlsx')
df_physics1 = pd.read_excel('../ground_truth_files/ground_truth_physics.xlsx')

row_name = new_row(row_name = 'Physics')

result_dfs = [df_physics2]
df_names = ['Consistency']

ground_truth_column = df_physics1['Slidenumber']

ground_truth_labels = ground_truth_column.unique()

for df, name in zip(result_dfs, df_names):
    result_column = df['Slidenumber']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1

df_physics2 = pd.read_excel('../ground_truth_files/team_dynamics_game_design_MIT_ground_truth_labeled_control.xlsx')
df_physics1 = pd.read_excel('../ground_truth_files/ground_truth_team_dynamics_game_design_MIT.xlsx')

row_name = new_row(row_name = 'Team dynamics')

result_dfs = [df_physics2]
df_names = ['Consistency']

ground_truth_column = df_physics1['Slidenumber']

ground_truth_labels = ground_truth_column.unique()

for df, name in zip(result_dfs, df_names):
    result_column = df['Slidenumber']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1

df_physics2 = pd.read_excel('../ground_truth_files/theory_of_computation_MIT_ground_truth_labeled_control.xlsx')
df_physics1 = pd.read_excel('../ground_truth_files/ground_truth_theory_of_computation_MIT.xlsx')

row_name = new_row(row_name = 'Theory of computation')

result_dfs = [df_physics2]
df_names = ['Consistency']

ground_truth_column = df_physics1['Slidenumber']

ground_truth_labels = ground_truth_column.unique()

for df, name in zip(result_dfs, df_names):
    result_column = df['Slidenumber']

    # Filter out rows where the ground truth -1 (meaning that no slide was shown but something else)
    mask = (ground_truth_column != -1) 
    filtered_ground_truth = ground_truth_column[mask]
    filtered_result = result_column[mask]

    # Calculate metrics
    precision = precision_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    recall = recall_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')
    f1 = f1_score(filtered_ground_truth, filtered_result, labels=ground_truth_labels, average='micro')

    df_precision.loc[row_name, name] = precision
    df_recall.loc[row_name, name] = recall
    df_F1.loc[row_name, name] = f1



df_precision.to_excel('results_precision_rater_consistency.xlsx', index=False, engine='openpyxl')
df_recall.to_excel('results_recall_rater_consistency.xlsx', index=False, engine='openpyxl')
df_F1.to_excel('results_F1_rater_consistency.xlsx', index=False, engine='openpyxl')