import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the csv files
loss_path = "/share/futianyu/repo/NLP-playground/local/universal/vicuna-7b-v1.5/q_long/output_pareto/opt_importance_loss.csv"
opt_importance_loss = pd.read_csv(loss_path, index_col=0)

# set the column to 1k, 2k, 4k
opt_importance_columns = ['1k_loss', '2k_loss', '3k_loss', '4k_loss']
# opt_importance_columns = ['1k_loss', '2k_loss', '4k_loss']
opt_importance_loss.columns = opt_importance_columns
test_summary = pd.read_csv('/share/futianyu/repo/qllm-eval/qllm_eval/evaluation/q_long/test_summary.csv')

# Pivot test_summary to have two columns for the two Test Dir
test_summary_pivot = test_summary.pivot(index='Plan ID', columns='Test Dir', values='Accuracy')

# Rename the columns of the pivoted DataFrame
test_summary_pivot.columns = list(test_summary_pivot.columns)

# Reset the index to make Plan ID a column
test_summary_pivot.reset_index(inplace=True)

# Merge the two dataframes based on the index of opt_importance_loss and Plan ID of test_summary_pivot
combined_df = pd.merge(test_summary_pivot, opt_importance_loss, left_on='Plan ID', right_index=True)
column_key_map = {
    '2k_cases': '2k_acc',
    '4k_cases': '4k_acc',
    '8k_cases': '8k_acc', 
    '15k_cases': '15k_acc'
}
# use current column names to filter column key map
column_key_map = {k: v for k, v in column_key_map.items() if k in combined_df.columns}
combined_df.rename(columns=column_key_map, inplace=True)

print(combined_df)

# save combined_df as csv
combined_df.to_csv("combined_df.csv", index=False)

"""
visualization
"""
import pandas as pd
import plotly.express as px

# Put the 15k_acc column at the last column
# remove the Plan ID column
combined_df = combined_df[[col for col in combined_df.columns if col != 'Plan ID']]
# move the 8k_acc and 15k_acc columns to the last column
combined_df = combined_df[opt_importance_columns + list(column_key_map.values())]

# Create the parallel coordinates plot
final_objective = "15k_acc"
fig = px.parallel_coordinates(combined_df,
                              color=final_objective,
                              dimensions=combined_df.columns,
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=combined_df[final_objective].mean())

# Save the figure as an HTML file
fig.write_html('parallel_coordinates.html')
# Save the figure as a PNG file
fig.write_image('parallel_coordinates.jpg')

print('done')