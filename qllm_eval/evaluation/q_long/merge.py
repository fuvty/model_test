import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the csv files
opt_importance_loss = pd.read_csv('/share/futianyu/repo/NLP-playground/local/universal/test-model/profile_test/opt_importance_loss.csv', index_col=0)

# set the column to 1k, 2k, 4k
opt_importance_loss.columns = ['1k_loss', '2k_loss', '4k_loss']

test_summary = pd.read_csv('/share/futianyu/repo/qllm-eval/qllm_eval/evaluation/q_long/test_summary.csv')

# Pivot test_summary to have two columns for the two Test Dir
test_summary_pivot = test_summary.pivot(index='Plan ID', columns='Test Dir', values='Accuracy')

# Rename the columns of the pivoted DataFrame
test_summary_pivot.columns = list(test_summary_pivot.columns)

# Reset the index to make Plan ID a column
test_summary_pivot.reset_index(inplace=True)

# Merge the two dataframes based on the index of opt_importance_loss and Plan ID of test_summary_pivot
combined_df = pd.merge(test_summary_pivot, opt_importance_loss, left_on='Plan ID', right_index=True)
combined_df.rename(columns={
    '8k_cases': '8k_acc', 
    '15k_cases': '15k_acc'
}, inplace=True)

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
combined_df = combined_df[['1k_loss', '2k_loss', '4k_loss', '8k_acc', '15k_acc']]

# Create the parallel coordinates plot
fig = px.parallel_coordinates(combined_df,
                              color="15k_acc",
                              dimensions=combined_df.columns,
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=combined_df['15k_acc'].mean())

# Save the figure as an HTML file
fig.write_html('parallel_coordinates.html')
# Save the figure as a PNG file
fig.write_image('parallel_coordinates.jpg')

print('done')