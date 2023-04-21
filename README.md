# Heart_Disease_Prediction
Took 
### About Heart disease
- Heart Disease (including Coronary Heart Disease, Hypertension, and Stroke) remains the No. 1 cause of death in the US. The Heart Disease and Stroke Statistics—2019 Update from the American Heart Association indicates that:

- 116.4 million, or 46% of US adults are estimated to have hypertension. These are findings related to the new 2017 Hypertension Clinical Practice Guidelines.
- On average, someone dies of CVD every 38 seconds. About 2,303 deaths from CVD each day, based on 2016 data.
- On average, someone dies of a stroke every 3.70 minutes. About 389.4 deaths from stroke each day, based on 2016 data.
### Project Overview
- In this project, we aim to use a machine learning ensemble technique to predict the presence of heart disease in patients.we will explore a machine learning ensemble approach to predict heart disease, as presented in the already available Github project. We will evaluate the existing models (e.g. Random Forest, Extra Tree, etc.) and also add the HRFLM (Hierarchical Random Forest Logistic Model) and CNN (Convolutional Neural Network) models to compare their results. This will allow us to assess the strengths and limitations of each model and choose the best one for predicting heart disease By combining the predictions of these models, we aim to improve the accuracy of our predictions and identify patients at high risk of heart disease.

### This Project is divided into 14 major steps which are as follows:

  1. Data description
  2. Importing Libraries & setting up environment
  3. Loading dataset
  4. Data Cleaning & Preprocessing
  5. Exploratory Data Analysis
  6. Outlier Detection & Removal
  7. Training & Test Split
  8. Cross Validation
  9. Model Building
 10. Model evaluation & comparison
 11. Feature Selection
 12. Model Evaluation
 13. HRFLM Model Implementation
 14. CNN Model Implementation
 15. Conclusion

### About Data
 - This dataset consists of 11 features and a target variable. It has 6 nominal variables and 5 numeric variables. The detailed description of all the features are as follows:

- Age: Patients Age in years (Numeric)
- Sex: Gender of the patient (Male - 1, Female - 0) (Nominal)
- Chest Pain Type: Type of chest pain experienced by the patient categorized into 1 typical, 2 typical angina, 3 non-anginal pain, 4 asymptomatic (Nominal)
- Resting bp s: Level of blood pressure at resting mode in mm/HG (Numerical)
- Cholesterol: Serum cholesterol in mg/dl (Numeric)
- Fasting blood sugar: Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false (Nominal)
- Resting ECG: Result of electrocardiogram while at rest are represented in 3 distinct values 0: Normal, 1: Abnormality in ST-T wave, 2: Left ventricular hypertrophy (Nominal)
- Max heart rate: Maximum heart rate achieved (Numeric)
- Exercise angina: Angina induced by exercise 0 depicting NO, 1 depicting Yes (Nominal)
- Oldpeak: Exercise-induced ST-depression in comparison with the state of rest (Numeric)
- ST slope: ST segment measured in terms of slope during peak exercise 0: Normal, 1: Upsloping, 2: Flat, 3: Downsloping (Nominal)

####Target variable
- Target: It is the target variable which we have to predict 1 means the patient is suffering from heart risk, and 0 means the patient is normal.

### Installations:
This project requires Python 3.x and the following Python libraries should be installed to get the project started:

- Numpy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
- Xgboost
- Keras
- Tensorflow
I also recommend installing Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project which also includes Jupyter Notebook to run and execute IPython Notebook.

#### Code :
Actual code to get started with the project is provided in two files one is,heart-disease-classification.ipynb

#### Run :
In a terminal or command window, navigate to the top-level project directory PIMA_Indian_Diabetes/ (that contains this README) and run one of the following commands:

      ipython notebook heart-disease-classification.ipynb or

      jupyter notebook heart-disease-classification.ipynb

This will open the Jupyter Notebook software and project file in your browser.

#### Model Evaluation :
   I have done model evaluation based on following sklearn metric.

- Cross Validation Score
- Confusion Matrix
- Plotting ROC-AUC Curve
- Plotting Precision recall Curve
- Sensitivity and Specitivity
- Classification Error
- Log Loss
- Mathew Correlation coefficient
