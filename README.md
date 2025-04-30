# Iris Dataset Analysis
Author: Carmine Giardino

## Project Description
This project focuses on analyzing the **Iris dataset**, a well-known dataset in the field of machine learning and statistics.  
The analysis includes generating summary statistics, visualizations, and insights into the dataset's features and species.  
The goal is to provide a comprehensive overview of the dataset, including variable summaries, histograms, and scatter plots for different variable pairs.  

## Project Plan
- **Week 1 (March 31st - April 6th):** Create the project plan and set up the repository.
- **Week 2 (April 7th - April 13th):** Research the Iris dataset and write a brief summary of it in the README.md file.
- **Week 3 (April 14th - April 20th):** Implement variable summary analysis and generate text output.
- **Week 4 (April 21st - April 27th):** Generate and save histograms for each variable.
- **Week 5 (April 28th - May 4th):** Create scatter plots for variable pairs and save them as images.
- **Week 6 (May 5th - May 11th):** Perform additional analysis and enhance documentation.
- **Week 7 (May 12th - May 12th):** Review, finalize, and submit the project repository.

## Iris Dataset Summary

The **Iris dataset**, introduced by **Ronald A. Fisher in 1936**, is commonly used for **classification and statistical analysis**. It contains **150 samples** divided equally among **three species**:

- **Setosa**
- **Versicolor**
- **Virginica**

### Features Measured
- **Sepal length (cm)**
- **Sepal width (cm)**
- **Petal length (cm)**
- **Petal width (cm)**

With **50 samples per species**, the dataset is popular for its simplicity, making it ideal for testing various data analytics and machine learning algorithms.

## How to Run the Project

### Prerequisites
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/CGiardino/pands-project.git
    cd pands-project
    ```
2. Install Python (version 3.7 or later).
3. Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
    ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook iris_analysis.ipynb
   ```
   Alternatively, you can run the Python script directly:
   ```bash
   python iris_analysis.py
   ```
   Note the python script is a conversion of the Jupyter Notebook.

### Expected Results
1. **Summary Statistics**: A text file named `iris_summary_statistics.txt` will be generated, containing descriptive statistics for each feature grouped by species.
2. **Histograms**: Histogram images for each feature (e.g., `sepal_length_histogram.png`).
3. **Scatter Plots**: Scatter plot images for each pair of features (e.g., `sepal_length_vs_sepal_width_scatter.png`).
4. **Pairplot**: A comprehensive pairplot image (`pairplot.png`).
5. **Correlation Matrix**: A heatmap image (`correlation_matrix.png`) showing feature correlations.
6. **Boxplots**: Boxplot images for each feature grouped by species (e.g., `sepal_length_boxplot.png`).

All results will be saved in the same directory as the script.

### Key Insights
- The **Setosa** species is generally smaller in size compared to **Versicolor** and **Virginica**.
- **Versicolor** and **Virginica** have overlapping features, making them harder to distinguish.
- The **petal length** and **width** are more discriminative features for species classification.