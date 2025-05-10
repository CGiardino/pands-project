# Iris Dataset Summary Statistics
# This project focuses on analyzing the Iris dataset, a well-known dataset in the field of machine learning and statistics.  
# The analysis includes generating summary statistics, visualizations, and insights into the dataset's features and species.  
# The goal is to provide a comprehensive overview of the dataset, including variable summaries, histograms, and scatter plots for different variable pairs.
# 
# Author: Carmine Giardino
# 
# Important Note: this is a copy of the analysis.ipynb, but without the display of the plots.

# Import pandas for data manipulation and analysis.
import pandas as pd
# Import load_iris to load the Iris dataset from scikit-learn.
from sklearn.datasets import load_iris
# Import matplotlib for plotting.
import matplotlib.pyplot as plt
# Import itertools for generating combinations of features.
import itertools
# Import seaborn for advanced visualizations
import seaborn as sns

# Loading the Iris dataset from scikit-learn and convert it to a pandas DataFrame.
# References:
# - [Load Iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
# - [Pandas Dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

# Load the Iris dataset.
iris = load_iris()

# Convert the dataset to a pandas DataFrame.
# Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the species names to the DataFrame
target_column_name = 'species'
# The target names are stored in the 'target' attribute of the dataset with integer labels. So we need to map them to the actual names.
iris_df[target_column_name] = iris.target_names[iris.target]

# Grouping the Iris DataFrame by the values in the column named "species". Therefore, separating groups for each class, e.g., setosa, versicolor, and virginica.  
# Describing each group with stats (count, mean, std, etc.), and flattening the group labels from the index into a column.
# References: 
# - [DataFrame groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)
# - [DataFrame describe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html)
# - [DataFrame rename axis](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename_axis.html)
# - [DataFrame reset index](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html)

iris_summary_by_class = iris_df.groupby(target_column_name).describe().reset_index()

# Joining multi-level column names tuples like ('sepal_length', 'mean') into a single string: 'sepal_length mean', making the DataFrame easier to work with.
# Reference: 
# - [Flatten headers](https://chatgpt.com/share/67f98414-0680-800f-aa4e-0e5b69996c99) - couldn't find an easy way to do this in the documentation.

iris_summary_by_class.columns = [' '.join(col).strip() for col in iris_summary_by_class.columns.values]

# Formatting the DataFrame summary as a string with 2 decimal places, and writing it to a text file. It combines saving output for records and showing results.
# Reference: 
# - [DataFrame to string](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_string.html)
# - [Python file write](https://www.w3schools.com/python/python_file_write.asp)

output_content = iris_summary_by_class.to_string(index=False, float_format='{:0.2f}'.format)
output_filename = 'iris_summary_statistics.txt'
with open(output_filename, 'w') as file:
    file.write("Variable Summary Analysis\n")
    file.write(output_content)
    file.write("\n")
    
# Creating histograms for each feature in the Iris dataset and save them as image files.  
# The histograms provide a visual representation of the distribution of each feature, allowing for easy identification of patterns and clusters.
# References:
# - [Matplotlib figure](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html)

# List of feature names
feature_names = iris.feature_names

# Create and save histograms for each feature
for feature in feature_names:
    plt.figure(figsize=(8, 6))

    # Plot histograms for each species without specifying colors
    for species in iris_df[target_column_name].unique():
        plt.hist(iris_df[iris_df[target_column_name] == species][feature], bins=20, alpha=0.6, label=species, edgecolor='black')

    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Species')

    # Save the histogram as an image file
    filename = feature.replace(' ', '_').replace('(', '').replace(')', '') + '_histogram.png'
    plt.savefig(filename)
    plt.close()

# Creating scatter plots for each pair of features in the Iris dataset and save them as image files.  
# The scatter plots provide a visual representation of the relationships between features.
# References:
# - [Matplotlib scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)
# - [Itertools combinations](https://docs.python.org/3/library/itertools.html#itertools.combinations)

# Create scatter plots for each pair of features
for feature_x, feature_y in itertools.combinations(feature_names, 2):
    plt.figure(figsize=(8, 6))

    # Plot scatter points for each species
    for species in iris_df['species'].unique():
        subset = iris_df[iris_df['species'] == species]
        plt.scatter(subset[feature_x], subset[feature_y], label=species, alpha=0.7)

    plt.title(f'Scatter Plot: {feature_x} vs {feature_y}')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(title='Species')
    plt.grid(linestyle='--', alpha=0.7)

    # Save the scatter plot as an image file
    filename = f"{feature_x.replace(' ', '_')}_vs_{feature_y.replace(' ', '_')}_scatter.png"
    plt.savefig(filename)
    plt.close()

# Creating a pairplot for the features in the Iris dataset and save it as an image file.  
# The pairplot provides a comprehensive view of the relationships between features, allowing for easy identification of patterns and clusters.
# References:
# - [Seaborn pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

# Create a pairplot for feature relationships
sns.pairplot(iris_df, hue='species', diag_kind='kde')
plt.savefig('pairplot.png')

# Creating a correlation matrix heatmap for the features in the Iris dataset and save it as an image file.  
# The correlation matrix provides a visual representation of the relationships between features, making it easier to identify strong correlations.
# References:
# - [Seaborn heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)

# Compute and visualize the correlation matrix
correlation_matrix = iris_df.iloc[:, :-1].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')

# Creating boxplots for each feature in the Iris dataset, grouped by species, and save them as image files.  
# The boxplots provide a visual representation of the distribution of each feature for different species, highlighting the differences and similarities between them.
# References:
# - [Seaborn boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html)
# Generate boxplots for each feature
for feature in feature_names:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='species', y=feature, data=iris_df)
    plt.title(f'Boxplot of {feature} by Species')
    plt.savefig(f'{feature.replace(" ", "_")}_boxplot.png')

# Key Insights
# - The Setosa species is generally smaller in size compared to Versicolor and Virginica.
# - Versicolor and Virginica have overlapping features, making them harder to distinguish.
# - The petal length and width are more discriminative features for species classification.

# Future Work
# As a next step, it would be useful to apply methods for predicting the species of flowers.
# For reference, **k-NN** and **Decision Trees** have already shown strong performance in preliminary tests:  
# - [Avuluri Venkatasaireddy’s article on Medium](https://medium.com/@avulurivenkatasaireddy/k-nearest-neighbors-and-implementation-on-iris-data-set-f5817dd33711) demonstrates the use of the k-NN algorithm with Scikit-Learn.  
# - [Wei-Lung Wang’s article on Medium](https://medium.com/@wl8380/understanding-decision-trees-with-the-iris-dataset-6c12d3b1b09d) shows how Decision Trees can be used to classify flowers using simple rules.
# Both methods gave similar accuracy, but Decision Trees have an extra advantage—they're easier to understand, quicker to make predictions, and show exactly how the decision was made in a way that's easy to follow.