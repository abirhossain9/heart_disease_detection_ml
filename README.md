<p align="center">
  <img width="250" height="280" src="images/nsulogo.png">
</p>                                        









  <h1 align="center">Project Name: Heart Disease Prediction Using Machine Learning</h1>
  <h2 align ="center">Course Number: CSE 445<br>
  Section: 04</br>
  Semester: Summer 2021</br><br>
  Faculty Name: Intisar Tahmid Naheen (ITN)</h2>
  <h3 align="center">Student Name: Tanjil Mahmud<br>
  Student ID: 1821458042<br>
  Email: tanjil.mahmud@northsouth.edu <br><br>
  <h3 align="center">Student Name: Md Abir Hossain<br>
  Student ID: 1731597042<br>
  Email: abir.hossain04@northsouth.edu <br><br>
  <h3 align="center">Student Name: Emamul Hassan<br>
  Student ID: 1731250042<br>
  Email: emamul.hassan@northsouth.edu <br><br>
  Date prepared: 16/09/2021</h3><br><br><br>
  
  # Heart Disease Prediction Using Machine Learning Approach:


Heart Disease (including Coronary Heart Disease, Hypertension, and Stroke) remains the No. 1
cause of death in the US.The Heart Disease and Stroke Statistics—2019 Update from the **American Heart Association** indicates that:
* 116.4 million, or 46% of US adults are estimated to have hypertension. These are findings related to the new 2017 Hypertension Clinical Practice Guidelines.
* On average, someone dies of CVD every 38 seconds. About 2,303 deaths from CVD each day, based on 2016 data.
* On average, someone dies of a stroke every 3.70 minutes. About 389.4 deaths from stroke each day, based on 2016 data.

In this notebook we will try to unleash useful insights using this heart disease datasets and by building stacked ensemble model by combining the power of best performing machine learning algorithms.

This notebook is divided into 13 major steps which are as follows:

1. [Data description](#data-desc)
2. [Importing Library](#lib-dataset)
3. Loading Dataset
4. Statistical Analysis of Dataset
5. Exploratory Data Analysis EDA (Categorical Variables)
6. Exploratory Data Analysis EDA (Numerical Variables)
7. Removing Outliers (Data Cleaning)
8. One Hot Encoding
9. Correlation Matrix
10. Training and Testing Data Spliting
11. Data Normalization
12. Model Building and Evaluation
13. ROC and AUC

## 1. Dataset description<a id='data-desc'></a>

This dataset consists of 11 features and a target variable. It has 6 nominal variables and 5 numeric variables. The detailed description of all the features are as follows:

**1. Age:** Patients Age in years (Numeric)<br>
**2. Sex:** Gender of patient (Male - 1, Female - 0) (Nominal)<br>
**3. Chest Pain Type:** Type of chest pain experienced by patient categorized into 1 typical, 2 typical angina, 3 non-        anginal pain, 4 asymptomatic (Nominal)<br>
**4. resting bp s:** Level of blood pressure at resting mode in mm/HG (Numerical)<br>
**5. cholestrol:** Serum cholestrol in mg/dl (Numeric)<br>
**6. fasting blood sugar:** Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false (Nominal)<br>
**7. resting ecg:** Result of electrocardiogram while at rest are represented in 3 distinct values 0 : Normal 1: Abnormality in ST-T wave 2: Left ventricular hypertrophy (Nominal)<br>
**8. max heart rate:** Maximum heart rate achieved (Numeric)<br>
**9. exercise angina:** Angina induced by exercise 0 depicting NO 1 depicting Yes (Nominal)<br>
**10. oldpeak:** Exercise induced ST-depression in comparison with the state of rest (Numeric)<br>
**11. ST slope:** ST segment measured in terms of slope during peak exercise 0: Normal 1: Upsloping 2: Flat 3: Downsloping (Nominal)<br>

#### Target variable
**12. target:** It is the target variable which we have to predict 1 means patient is suffering from heart risk and 0 means patient is normal.


## 2. Importing Library<a id='lib-dataset'></a>    

import warnings<br>
warnings.filterwarnings('ignore') <br>

import pandas as pd<br>
import numpy as np<br>

import matplotlib.pyplot as plt<br>
%matplotlib inline<br>
import seaborn as sns<br><br>
from scipy import stats<br>

import seaborn as sns<br>

from sklearn.model_selection import train_test_split

### model validation
from sklearn.metrics import log_loss,roc_auc_score,precision_score,f1_score,recall_score,roc_curve,auc<br>
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,fbeta_score,matthews_corrcoef<br>
from sklearn import metrics

### cross validation
from sklearn.model_selection import StratifiedKFold<br>

### machine learning algorithms
from sklearn.linear_model import LogisticRegression<br>
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier<br>
from sklearn.neural_network import MLPClassifier<br>
from sklearn.tree import DecisionTreeClassifier<br>
from sklearn.linear_model import SGDClassifier<br>
from sklearn.svm import SVC<br>

from sklearn.preprocessing import MinMaxScaler<br>

from sklearn import model_selection<br>
from sklearn.model_selection import cross_val_score<br>

from sklearn.naive_bayes import GaussianNB<br>
from sklearn.metrics import roc_curve, roc_auc_score<br>

## Note: For Viewing Steps [3-13], please click the link given below.

### Link: [Remaining Steps](https://github.com/EmamulHassan/heart_disease_detection_ml/blob/main/Project%20445%20demo.ipynb)
    
    
