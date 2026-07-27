<H3>SADHANA S</H3>
<H3>212224230234</H3>
<H3>EX. NO.1</H3>
<H3>27.7.26</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\admin\Downloads\2026_team_summaries.csv.xls")
df

df.isnull().sum()

df = df.fillna(0)
df.isnull().sum()

df.duplicated().sum()

df.info()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df.head()

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
print("X Values")
print(X)
print("Y Values")
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print("X Training Data")
print(X_train)
print("X Testing Data")
print(X_test)
print("Y Training Data")
print(Y_train)
print("Y Testing Data")
print(Y_test)

```

## OUTPUT:

Read the Dataset

<img width="1068" height="817" alt="image" src="https://github.com/user-attachments/assets/48d714f1-9b0d-4d84-9f8e-05eadcea7671" />


Check Missing Values

<img width="542" height="141" alt="image" src="https://github.com/user-attachments/assets/39b3f033-ed10-4dc6-8edf-7cc3c5ba15c8" />


Fill Missing Values

<img width="497" height="150" alt="image" src="https://github.com/user-attachments/assets/bf17e582-84ef-44a4-9601-6e301538f663" />


Check Duplicate Rows

<img width="383" height="90" alt="image" src="https://github.com/user-attachments/assets/24d70d53-41a5-4a04-a71f-be05269c8f2e" />


Display Dataset Information

<img width="655" height="251" alt="image" src="https://github.com/user-attachments/assets/bf095c6a-8448-479e-b10e-52876048e231" />


Normalize Numeric Columns

<img width="1002" height="218" alt="image" src="https://github.com/user-attachments/assets/dd05573d-466d-4e00-8d09-4a0e2d40b5a7" />


Split into Input and Output

<img width="1227" height="377" alt="image" src="https://github.com/user-attachments/assets/8cce2829-eb3f-4589-8a62-fe0a183f9b46" />


<img width="1237" height="377" alt="image" src="https://github.com/user-attachments/assets/044b99f4-8dc8-4763-b9ad-8587e91b7c60" />



Train-Test Split

<img width="1248" height="383" alt="image" src="https://github.com/user-attachments/assets/caff4835-1dd7-4ad9-8116-7c60e6eea711" />


<img width="1246" height="376" alt="image" src="https://github.com/user-attachments/assets/8b64f353-c230-45d7-8725-3d09158a099d" />




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


