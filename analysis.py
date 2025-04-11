# Author: Carmine Giardino

# Import pandas for data manipulation and analysis.
import pandas as pd
# Import load_iris to load the Iris dataset from scikit-learn.
from sklearn.datasets import load_iris

# Load the Iris dataset.
iris = load_iris()

# Convert the dataset to a pandas DataFrame.
# Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the species names to the DataFrame
target_column_name = 'species'
# The target names are stored in the 'target' attribute of the dataset with integer labels. So we need to map them to the actual names.
iris_df[target_column_name] = iris.target_names[iris.target]

# Group by species name and compute summary statistics.
# References: 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename_axis.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
iris_summary_by_class = iris_df.groupby(target_column_name).describe().reset_index()

# Flatten the multi-level column headers.
# Reference: https://chatgpt.com/share/67f98414-0680-800f-aa4e-0e5b69996c99
# Couldn't find an easy way to do this in the documentation.
iris_summary_by_class.columns = [' '.join(col).strip() for col in iris_summary_by_class.columns.values]

# Save the summary DataFrame to a text file.
# Reference: 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_string.html
# https://www.w3schools.com/python/python_file_write.asp
output_filename = 'iris_summary_statistics.txt'
with open(output_filename, 'w') as file:
    file.write("Variable Summary Analysis\n")
    file.write(iris_summary_by_class.to_string(index=False, float_format='{:0.2f}'.format))
    file.write("\n")