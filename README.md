# Air_quality_index_predictor
Air Quality Index Predictor project is to predict the AQI level. 
````markdown
# Air Quality Index Predictor

## Project Overview

This project provides a comprehensive solution for predicting the Air Quality Index (AQI) based on various air pollutant concentrations and weather conditions.
 It includes two distinct models built using Artificial Neural Networks (ANN): a **regression model** to predict the precise numerical AQI value
and a **classification model** to predict the corresponding AQI category (e.g., "Good", "Unhealthy").

The project utilizes the `global_air_quality_dataset` from Kaggle, which contains a rich set of features including:

-   **Pollutants**: PM2.5, PM10, NO2, SO2, CO, O3
-   **Weather Data**: Temperature, Humidity, Wind Speed

The solution is structured into several key phases:

1.  **Data Loading & Inspection**: Loading the dataset and performing an initial check for shape, columns, and missing values.
2.  **Preprocessing**: Cleaning the data, calculating the target AQI values based on US EPA guidelines, and scaling the features for optimal model performance.
3.  **Model Building**: Developing two separate ANN models—one for regression and one for classification.
4.  **Training**: Compiling and training the models using an efficient optimizer and callbacks to prevent overfitting.
5.  **Evaluation**: Assessing model performance with appropriate metrics for both regression (MAE, RMSE, $R^2$) and classification (Accuracy, Precision, Recall, F1-score, Confusion Matrix).
6.  **Inference**: Providing a snippet for making real-time predictions on a single new data point.

---


````

  - **pandas**: For data manipulation and analysis.
  - **numpy**: For numerical operations, especially with arrays.
  - **scikit-learn**: For data splitting, preprocessing (MinMaxScaler, LabelEncoder), and evaluation metrics.
  - **tensorflow** and **Keras**: For building, training, and evaluating the neural network models.
  - **matplotlib** and **seaborn**: For data visualization, including plots and heatmaps.

-----

## How to Run the Project

1.  **Download the Dataset**: Ensure the `global_air_quality_data.csv` file is in the same directory as the notebook. The notebook is set to look for this file at `DATA_PATH = "global_air_quality_data.csv"`.

2.  **Open and Run the Notebook**: Open the `Air_Quality_Index_Predictor_combined.ipynb` file in a Jupyter Notebook or Google Colab environment.

3.  **Execute Cells**: Run all the cells in the notebook sequentially. The notebook is designed to be executed from top to bottom, with each cell building on the output of the previous one.

4.  **Review the Output**: The notebook will print model summaries, training history, evaluation metrics, and visualizations. The final cells demonstrate how to make a real-time prediction using the trained models.

### Key Outputs to Look For

  - **Model Summaries**: Details on the layers and parameters of both the regression and classification ANNs.
  - **Training Plots**: Visualizations of training and validation loss/accuracy to show how the models learned over epochs.
  - **Evaluation Metrics**:
      - **Regression**: MAE, RMSE, and $R^2$ scores, which quantify the model's predictive accuracy for numerical AQI.
      - **Classification**: Accuracy, a detailed classification report, and a confusion matrix to show the model's performance in predicting AQI categories.
  - **Real-time Prediction**: A simple function and example usage to predict the AQI for a new set of pollutant and weather readings.

<!-- end list -->

```
```
