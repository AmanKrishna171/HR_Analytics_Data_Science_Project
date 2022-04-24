         


# IBM Attrition Data Analysis 
Done by : Aman Krishna Satheesh


## Inspiration for choosing the dataset 
My group's topic for last year's software engineering was people analytics in companies so I chose IBM's employee dataset with attrition value. 
Attrition also known as churn rate is the rate at which people leave the company. So companies use people analytics to find reasons to increase productive by exploring their employee's data.

I am predicting attrition based on other features like age, gender, monthly income etc.

The dataset is from [here](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

All the imports


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix,roc_auc_score,plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,accuracy_score
from pycm import ConfusionMatrix
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.naive_bayes import GaussianNB
```

Importing Dataset as pandas dataframe


```python
df = pd.read_csv(r"")
df.dataframeName = 'IBM.csv'
nRow, nCol = df.shape
print(f'{nRow} rows and {nCol} columns')
```

    1470 rows and 35 columns


## Exploring the Dataset


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>...</td>
      <td>1470.000000</td>
      <td>1470.0</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
      <td>1470.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.923810</td>
      <td>802.485714</td>
      <td>9.192517</td>
      <td>2.912925</td>
      <td>1.0</td>
      <td>1024.865306</td>
      <td>2.721769</td>
      <td>65.891156</td>
      <td>2.729932</td>
      <td>2.063946</td>
      <td>...</td>
      <td>2.712245</td>
      <td>80.0</td>
      <td>0.793878</td>
      <td>11.279592</td>
      <td>2.799320</td>
      <td>2.761224</td>
      <td>7.008163</td>
      <td>4.229252</td>
      <td>2.187755</td>
      <td>4.123129</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.135373</td>
      <td>403.509100</td>
      <td>8.106864</td>
      <td>1.024165</td>
      <td>0.0</td>
      <td>602.024335</td>
      <td>1.093082</td>
      <td>20.329428</td>
      <td>0.711561</td>
      <td>1.106940</td>
      <td>...</td>
      <td>1.081209</td>
      <td>0.0</td>
      <td>0.852077</td>
      <td>7.780782</td>
      <td>1.289271</td>
      <td>0.706476</td>
      <td>6.126525</td>
      <td>3.623137</td>
      <td>3.222430</td>
      <td>3.568136</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>465.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
      <td>491.250000</td>
      <td>2.000000</td>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.000000</td>
      <td>802.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>1020.500000</td>
      <td>3.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>1157.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>1555.750000</td>
      <td>4.000000</td>
      <td>83.750000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.000000</td>
      <td>1499.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>2068.000000</td>
      <td>4.000000</td>
      <td>100.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>



## Checking for incomplete entries and datatypes


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   Age                       1470 non-null   int64 
     1   Attrition                 1470 non-null   object
     2   BusinessTravel            1470 non-null   object
     3   DailyRate                 1470 non-null   int64 
     4   Department                1470 non-null   object
     5   DistanceFromHome          1470 non-null   int64 
     6   Education                 1470 non-null   int64 
     7   EducationField            1470 non-null   object
     8   EmployeeCount             1470 non-null   int64 
     9   EmployeeNumber            1470 non-null   int64 
     10  EnvironmentSatisfaction   1470 non-null   int64 
     11  Gender                    1470 non-null   object
     12  HourlyRate                1470 non-null   int64 
     13  JobInvolvement            1470 non-null   int64 
     14  JobLevel                  1470 non-null   int64 
     15  JobRole                   1470 non-null   object
     16  JobSatisfaction           1470 non-null   int64 
     17  MaritalStatus             1470 non-null   object
     18  MonthlyIncome             1470 non-null   int64 
     19  MonthlyRate               1470 non-null   int64 
     20  NumCompaniesWorked        1470 non-null   int64 
     21  Over18                    1470 non-null   object
     22  OverTime                  1470 non-null   object
     23  PercentSalaryHike         1470 non-null   int64 
     24  PerformanceRating         1470 non-null   int64 
     25  RelationshipSatisfaction  1470 non-null   int64 
     26  StandardHours             1470 non-null   int64 
     27  StockOptionLevel          1470 non-null   int64 
     28  TotalWorkingYears         1470 non-null   int64 
     29  TrainingTimesLastYear     1470 non-null   int64 
     30  WorkLifeBalance           1470 non-null   int64 
     31  YearsAtCompany            1470 non-null   int64 
     32  YearsInCurrentRole        1470 non-null   int64 
     33  YearsSinceLastPromotion   1470 non-null   int64 
     34  YearsWithCurrManager      1470 non-null   int64 
    dtypes: int64(26), object(9)
    memory usage: 402.1+ KB


There are no null or incomplete values

# Data Exploration 


```python
categorical_data = df.select_dtypes('object')
categorical_data.keys()
```




    Index(['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender',
           'JobRole', 'MaritalStatus', 'Over18', 'OverTime'],
          dtype='object')



Therefore there are 9 features with non numeric values

## Their values


```python
for i in categorical_data.keys():
    print(i," : ",df[i].unique(),"\n")
```

    Attrition  :  ['Yes' 'No'] 
    
    BusinessTravel  :  ['Travel_Rarely' 'Travel_Frequently' 'Non-Travel'] 
    
    Department  :  ['Sales' 'Research & Development' 'Human Resources'] 
    
    EducationField  :  ['Life Sciences' 'Other' 'Medical' 'Marketing' 'Technical Degree'
     'Human Resources'] 
    
    Gender  :  ['Female' 'Male'] 
    
    JobRole  :  ['Sales Executive' 'Research Scientist' 'Laboratory Technician'
     'Manufacturing Director' 'Healthcare Representative' 'Manager'
     'Sales Representative' 'Research Director' 'Human Resources'] 
    
    MaritalStatus  :  ['Single' 'Married' 'Divorced'] 
    
    Over18  :  ['Y'] 
    
    OverTime  :  ['Yes' 'No'] 
    



```python
df.columns[df.nunique() <= 1]
```




    Index(['EmployeeCount', 'Over18', 'StandardHours'], dtype='object')



since the above features only have a single value, it is better to drop them


```python
df.drop('EmployeeCount',axis =1,inplace=True)
df.drop('StandardHours',axis =1,inplace=True)
df.drop('Over18',axis =1,inplace=True)
df.drop('PerformanceRating',axis =1,inplace=True)

```

We can also see EmployeeNumber which would also not help in any form of predicition so that can be dropped as well 


```python
df.drop('EmployeeNumber',axis =1,inplace=True)
```

# Data Visualization 


```python
sns.set(rc = {'figure.figsize':(25,10)})
sns.histplot(data = df,
            x = "Age",
            hue = "Attrition",
            palette = "rainbow").set()
```




    []




    
![png](/images/output_22_1.png)
    


Here is a age 


```python
sns.countplot(x='BusinessTravel', data=df, hue="Attrition")


```




    <AxesSubplot:xlabel='BusinessTravel', ylabel='count'>




    
![png](/images/output_24_1.png)
    



```python
sns.histplot(data = df,
            x = "JobLevel",
            hue = "Attrition",
            palette = "autumn").set()

```




    []




    
![png](/images/output_25_1.png)
    



```python
sns.histplot(data = df,
            x = "Department",
            hue = "Attrition",
            palette = "autumn").set()
```




    []




    
![png](/images/output_26_1.png)
    



```python
sns.set(rc = {'figure.figsize':(30,10)})
sns.histplot(data = df,
            x = "JobRole",
            hue = "Attrition",
            palette = "autumn").set()
```




    []




    
![png](/images/output_27_1.png)
    



```python
sns.set(rc = {'figure.figsize':(10,10)})
sns.histplot(data = df,
            x = "Gender",
            hue = "Attrition",
            palette = "autumn").set()
```




    []




    
![png](/images/output_28_1.png)
    



```python
sns.histplot(data = df,
            x = "PercentSalaryHike",
            hue = "Attrition",
            palette = "autumn").set()
```




    []




    
![png](/images/output_29_1.png)
    



```python

sns.histplot(data = df,
            x = "BusinessTravel",
            hue = "Attrition",
            palette = "autumn").set()
```




    []




    
![png](/images/output_30_1.png)
    



```python
sns.histplot(data = df,
            x = "Education",
            hue = "Attrition",
            palette = "autumn").set()
```




    []




    
![png](/images/output_31_1.png)
    



```python
sns.histplot(data = df,
            x = "MaritalStatus",
            hue = "Attrition",
            palette = "autumn").set()
```




    []




    
![png](/images/output_32_1.png)
    


## Key Observations

1) More males have left the company than females
2) Employees who travel rarely have stayed in the company
3) More employees of Research & Development department have stayed in the company 
4) Married employees stay more often than single
5) More employees who work more overtime leave the company comapred to people who dont.
6) Employees with low Monthly income and low Job Level leave the company than those with high. 

# Data Preprocessing

## Checking for Skewness 


```python
df.skew()
```

    C:\Users\amank\AppData\Local\Temp/ipykernel_10996/1665899112.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      df.skew()





    Age                         0.413286
    DailyRate                  -0.003519
    DistanceFromHome            0.958118
    Education                  -0.289681
    EnvironmentSatisfaction    -0.321654
    HourlyRate                 -0.032311
    JobInvolvement             -0.498419
    JobLevel                    1.025401
    JobSatisfaction            -0.329672
    MonthlyIncome               1.369817
    MonthlyRate                 0.018578
    NumCompaniesWorked          1.026471
    PercentSalaryHike           0.821128
    RelationshipSatisfaction   -0.302828
    StockOptionLevel            0.968980
    TotalWorkingYears           1.117172
    TrainingTimesLastYear       0.553124
    WorkLifeBalance            -0.552480
    YearsAtCompany              1.764529
    YearsInCurrentRole          0.917363
    YearsSinceLastPromotion     1.984290
    YearsWithCurrManager        0.833451
    dtype: float64




```python
df.hist(figsize=(18,18),grid=True,bins='auto');
```


    
![png](/images/output_37_0.png)
    


We use log function to take care of the skewness for columns with greater than 0.5 skew score


```python
for index in df.skew().index:
    if (df.skew().loc[index]>0.5 or df.skew().loc[index]<-0.5):
        df[index]=np.log1p(df[index])

df.hist(figsize=(18,18),grid=True,bins='auto');        
```

    C:\Users\amank\AppData\Local\Temp/ipykernel_10996/4191402575.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      for index in df.skew().index:
    C:\Users\amank\AppData\Local\Temp/ipykernel_10996/4191402575.py:2: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      if (df.skew().loc[index]>0.5 or df.skew().loc[index]<-0.5):



    
![png](/images/output_39_1.png)
    


# Removing /images/outliers 
## Using zscore

Reference : https://machinelearningmastery.com/model-based-/images/outlier-detection-and-removal-in-python/


```python
df1=df.select_dtypes(exclude=['object'])
for column in df1:
        plt.figure(figsize=(5,5))
        sns.boxplot(data=df1, x=column)
```

    C:\Users\amank\AppData\Local\Temp/ipykernel_10996/1300572045.py:3: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      plt.figure(figsize=(5,5))



    
![png](/images/output_41_1.png)
    



    
![png](/images/output_41_2.png)
    



    
![png](/images/output_41_3.png)
    



    
![png](/images/output_41_4.png)
    



    
![png](/images/output_41_5.png)
    



    
![png](/images/output_41_6.png)
    



    
![png](/images/output_41_7.png)
    



    
![png](/images/output_41_8.png)
    



    
![png](/images/output_41_9.png)
    



    
![png](/images/output_41_10.png)
    



    
![png](/images/output_41_11.png)
    



    
![png](/images/output_41_12.png)
    



    
![png](/images/output_41_13.png)
    



    
![png](/images/output_41_14.png)
    



    
![png](/images/output_41_15.png)
    



    
![png](/images/output_41_16.png)
    



    
![png](/images/output_41_17.png)
    



    
![png](/images/output_41_18.png)
    



    
![png](/images/output_41_19.png)
    



    
![png](/images/output_41_20.png)
    



    
![png](/images/output_41_21.png)
    



    
![png](/images/output_41_22.png)
    


Now using zscore to remove to /images/outliers 


```python
num_train = df.select_dtypes(include=["number"])
cat_train = df.select_dtypes(exclude=["number"])
idx = np.all(stats.zscore(num_train) < 3, axis=1)
df_clean = pd.concat([num_train.loc[idx], cat_train.loc[idx]], axis=1) 
# reference : https://stackoverflow.com/questions/54398554/how-to-remove-/images/outliers-in-python

df1=df_clean.select_dtypes(exclude=['object'])
for column in df1:
        plt.figure(figsize=(5,5))
        sns.boxplot(data=df1, x=column)
```

    C:\Users\amank\AppData\Local\Temp/ipykernel_10996/2476272619.py:9: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      plt.figure(figsize=(5,5))



    
![png](/images/output_43_1.png)
    



    
![png](/images/output_43_2.png)
    



    
![png](/images/output_43_3.png)
    



    
![png](/images/output_43_4.png)
    



    
![png](/images/output_43_5.png)
    



    
![png](/images/output_43_6.png)
    



    
![png](/images/output_43_7.png)
    



    
![png](/images/output_43_8.png)
    



    
![png](/images/output_43_9.png)
    



    
![png](/images/output_43_10.png)
    



    
![png](/images/output_43_11.png)
    



    
![png](/images/output_43_12.png)
    



    
![png](/images/output_43_13.png)
    



    
![png](/images/output_43_14.png)
    



    
![png](/images/output_43_15.png)
    



    
![png](/images/output_43_16.png)
    



    
![png](/images/output_43_17.png)
    



    
![png](/images/output_43_18.png)
    



    
![png](/images/output_43_19.png)
    



    
![png](/images/output_43_20.png)
    



    
![png](/images/output_43_21.png)
    



    
![png](/images/output_43_22.png)
    


## Converting categorical values to int for classifier


```python
df.replace({'Yes':1,'No':0,'Travel_Rarely' : 1, 'Travel_Frequently' : 2, 'Non-Travel' : 3,
'Sales' : 1, 'Research & Development' : 2, 'Human Resources' : 3}, inplace=True)
X = df.drop(['Attrition'],axis=1) # attrition is dropped as it is the feature being predicted
```

# Correlation Matrix


```python
corr_matrix = df.corr()

plt.figure(figsize=(30,35))
sns.set(font_scale = 1.1)
sns.heatmap(corr_matrix, cmap='RdGy_r', annot=True, fmt='.2f')
plt.show()

```


    
![png](/images/output_47_0.png)
    


# Manual Feature Selection

From the corelation matrix, we can remove the less correlated features with attrition(target class) from the dataset.


```python
df.drop('NumCompaniesWorked',axis =1,inplace=True)
df.drop('TrainingTimeLastYear',axis =1,inplace=True)
```

## Splitting the data for classifying


```python
Y = df['Attrition']
# encoder.fit(X) 
X = pd.get_dummies(X)
# X = pd.get_dummies(X)
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, shuffle= True)

```

# Gaussian Naive Bayes

reference : https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/


```python
GaussianNB_classifier = GaussianNB()
y_pred = GaussianNB_classifier.fit(X_train, y_train).predict(X_test) #running the classifier

print("Number of mislabeled points /images/out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
```

    Number of mislabeled points /images/out of a total 294 points : 67


## Confusion Matrix 
It is generated to get the required metrics from the classifier


```python
cm = ConfusionMatrix(actual_vector=np.array(y_test), predict_vector=np.array(y_pred))

TPR = cm.class_stat.get("TPR")
FPR = cm.class_stat.get("FPR")
TNR = cm.class_stat.get("TNR")
FNR  = cm.class_stat.get("FNR")
ACC = cm.class_stat.get("ACC")
F1 = cm.class_stat.get("F1")
AUC= cm.class_stat.get("AUC")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_area = roc_auc_score(y_test, GaussianNB_classifier.predict_proba(X_test)[:,1])
```

# Major Metrics from Naïve Bayes Classifier 


```python
print("Accuracy : ", ACC)
print("Precision : ", precision)
print("Recall : ", recall)
print("True Positive Rate : ", TPR)
print("False Positive Rate : ", FPR)
print("True Negative Rate : ", TNR)
print("False Negative Rate : ", FNR)
print("F1 Score : ", F1)
print("ROC Area : ", AUC)
```

    Accuracy :  {0: 0.7721088435374149, 1: 0.7721088435374149}
    Precision :  0.31746031746031744
    Recall :  0.45454545454545453
    True Positive Rate :  {0: 0.828, 1: 0.45454545454545453}
    False Positive Rate :  {0: 0.5454545454545454, 1: 0.17200000000000004}
    True Negative Rate :  {0: 0.45454545454545453, 1: 0.828}
    False Negative Rate :  {0: 0.17200000000000004, 1: 0.5454545454545454}
    F1 Score :  {0: 0.8607068607068608, 1: 0.37383177570093457}
    ROC Area :  {0: 0.6412727272727272, 1: 0.6412727272727272}


## ROC Curve 


```python
probs = GaussianNB_classifier.predict_proba(X_test)
preds = probs[:,1]
fpr = dict()
tpr = dict()
roc_auc = dict()

# calculate dummies once
y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# roc for each class
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
for i in range(2):
    ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
ax.legend(loc="best")
ax.grid(alpha=.4)
sns.despine()
plt.show()




```


    
![png](/images/output_60_0.png)
    


# Feature selection 
## is used find the three sub datasets that contain more (combined features) or less (features removed). 

### Scoring argument specifies the evaluation criterion to be used. For classifier accuracy is used

The below function is used to extract data after sequential feature selection


```python
def get_features(results):
    max_score = max(results.avg_score)
    count = 0
    max_index = 0
    result = []
    for i in results.avg_score:
        count+=1
        if (i == max_score) :
            max_index = count
            result.append(results.feature_names[max_index])
            result.append(results.avg_score[max_index])
            break

    return result
    
```

## Forward selection 
starts with one predictor and adds more iteratively. At each subsequent iteration, the best of the remaining original predictors are added based on performance criteria.

Reference : https://towardsdatascience.com/feature-selection-for-machine-learning-in-python-wrapper-methods-2b5e27d2db31


```python
GaussianNB_classifier  = GaussianNB()


sfs = SFS(GaussianNB(),
           k_features=30,
           forward=True,
           floating=False,
           scoring = 'accuracy',
           cv = 5,
           n_jobs=-1
           )

sfs.fit(X_train, y_train)


df_SFS_results_forward_selection = pd.DataFrame(sfs.subsets_).transpose()

Highest_SFS_results_forward_selection = get_features(df_SFS_results_forward_selection)


fig = plot_sfs(sfs.get_metric_dict(), kind='std_err',figsize=(20,10))
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


```


    
![png](/images/output_66_0.png)
    


## Backward elimination 
starts with all predictors and eliminates one-by-one iteratively. One of the most popular algorithms is Recursive Feature Elimination (RFE) which eliminates less important predictors based on feature importance ranking.

Reference : https://towardsdatascience.com/feature-selection-for-machine-learning-in-python-wrapper-methods-2b5e27d2db31


```python
sfs = SFS(GaussianNB(),
           k_features=2, #here 2 is the minimum number of features it will start with 
           forward=False,
           floating=False,
           scoring = 'accuracy',
           cv = 5,
           n_jobs=-1)
#Use SFS to select the top 5 features 
sfs.fit(X_train, y_train)

#Create a dataframe for the SFS results 
df_SFS_results_backward_selection = pd.DataFrame(sfs.subsets_).transpose()
Highest_SFS_results_backward_selection = get_features(df_SFS_results_backward_selection)

fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_err',figsize=(20,10))
plt.title('Sequential Backward Selection (w. StdErr)')
plt.grid()
plt.show()
```


    
![png](/images/output_68_0.png)
    


## Step-wise selection 
bi-directional, based on a combination of forward selection and backward elimination. It is considered less greedy than the previous two procedures since it does reconsider adding predictors back into the model that has been removed (and vice versa). Nonetheless, the considerations are still made based on local optimisation at any given iteration.

Reference : https://towardsdatascience.com/feature-selection-for-machine-learning-in-python-wrapper-methods-2b5e27d2db31


```python
sfs = SFS(GaussianNB(),
           k_features=30,
           forward=True,
           floating=True,
           scoring = 'accuracy',
           cv = 5,
           n_jobs=-1)
 
sfs.fit(X_train, y_train)

#Create a dataframe for the SFS results 
df_SFS_results_BiDirectional_selection = pd.DataFrame(sfs.subsets_).transpose()
Highest_SFS_results_BiDirectional_selection = get_features(df_SFS_results_BiDirectional_selection)

fig = plot_sfs(sfs.get_metric_dict(), kind='std_err',figsize=(20,10))

plt.title('Sequential Step-wise Selection (w. StdErr)')
plt.grid()
plt.show()
```


    
![png](/images/output_70_0.png)
    



```python
print("Highest Accuracy obtained from forward selection : ", Highest_SFS_results_forward_selection[1])
print("\n")
print("List of best features obtained from forwards selection : ", Highest_SFS_results_forward_selection[0])
print("==============================================================================================================================================\n\n")

print("Highest Accuracy obtained from backward selection : ", Highest_SFS_results_backward_selection[1])
print("\n")
print("List of best features obtained from backward selection : ", Highest_SFS_results_backward_selection[0])
print("==============================================================================================================================================\n\n")

print("Highest Accuracy obtained from Bi-Directional selection : ", Highest_SFS_results_BiDirectional_selection[1])
print("\n")
print("List of best features obtained from Bi-Directional selection : ", Highest_SFS_results_BiDirectional_selection[0])
print("==============================================================================================================================================\n\n")
```

    Highest Accuracy obtained from forward selection :  0.8792643346556076
    
    
    List of best features obtained from forwards selection :  ('DailyRate', 'Department', 'DistanceFromHome', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'NumCompaniesWorked', 'OverTime', 'TotalWorkingYears', 'WorkLifeBalance', 'YearsSinceLastPromotion', 'EducationField_Life Sciences', 'EducationField_Medical', 'JobRole_Research Scientist', 'JobRole_Sales Executive', 'MaritalStatus_Single')
    ==============================================================================================================================================
    
    
    Highest Accuracy obtained from backward selection :  0.873306887847097
    
    
    List of best features obtained from backward selection :  ('DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'OverTime', 'RelationshipSatisfaction', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsInCurrentRole', 'EducationField_3', 'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Technical Degree', 'JobRole_Healthcare Representative', 'JobRole_Laboratory Technician', 'MaritalStatus_Single')
    ==============================================================================================================================================
    
    
    Highest Accuracy obtained from Bi-Directional selection :  0.8818175261449694
    
    
    List of best features obtained from Bi-Directional selection :  ('Department', 'DistanceFromHome', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'NumCompaniesWorked', 'OverTime', 'TotalWorkingYears', 'WorkLifeBalance', 'YearsSinceLastPromotion', 'EducationField_Life Sciences', 'EducationField_Medical', 'MaritalStatus_Single')
    ==============================================================================================================================================
    
    


## The best accuracy obtained from "bi direction feature selection" gave an accuracy of 87%
## Therefore, the accuracy increased from 78% to 87% with feature selection.  

# Conclusion:

I found it very interesting to do data exploration on this dataset. I found many correlations between age, job role, marital status, working hours business travel and the chances of the employee's staying in the company. I found many /images/outliers in monthly income and working hours which indicate that people were being overworked and also being paid less and thus have high attrition.
It is common sense but seeing data support it was very interesting. 

The least reliable feature was job satisfaction cause that can depend on a lot of other factors so correlation was less. 

The use of feature selection was also very beneficial. I used 3 methods and each of them got a higher accuracy than using the regular (pre-processed) dataset.  
I also did some experiments of my own :
1) Using dataset with/images/out removing skewness and /images/outliers : This gave an accuracy of around 68%
2) Using dataset with different types of methods in removing /images/outliers : I found using zscore was the most effective.


# Part - 2 : Clustering 


```python
# Imports for Clustering 

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib as mp
from sklearn import metrics
from sklearn import metrics 
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans,AffinityPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,davies_bouldin_score
from kneed import KneeLocator

```

## K-Means Clustering


### Use Elbow method to find the right number of clusters 

TO find the elbow


```python
def Knee(rng, data):
    kl = KneeLocator(    rng, data, curve="convex", direction="decreasing" )
    return kl.elbow
```


```python
Inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) 
    #k-means++ is used because it is recommended in documentation as it helps converge faster
    kmeans.fit(df)
    Inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 11), Inertia, 'bx-')
plt.title('Elbow Method',fontsize=14)
plt.xlabel('Number of clusters',fontsize=14)
plt.ylabel('Inertia',fontsize=14)
plt.show()

print("Elbow is at ",Knee(range(1,11),Inertia))
```


    
![png](/images/output_81_0.png)
    


    Elbow is at  3


From the elbow method, the optimal number of clusters is between 3. 

## Using Silhouette analysis to get optimal number of clusters 


```python
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
y=Y
range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
plt.show()

#code modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
```

    For n_clusters = 2 The average silhouette_score is : 0.4662185833987485
    For n_clusters = 3 The average silhouette_score is : 0.4930177608881641
    For n_clusters = 4 The average silhouette_score is : 0.41708911882831895
    For n_clusters = 5 The average silhouette_score is : 0.4348960794061264
    For n_clusters = 6 The average silhouette_score is : 0.38623990925174945



    
![png](/images/output_84_1.png)
    



    
![png](/images/output_84_2.png)
    



    
![png](/images/output_84_3.png)
    



    
![png](/images/output_84_4.png)
    



    
![png](/images/output_84_5.png)
    



```python
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
 
 # initialise kmeans
 kmeans = KMeans(n_clusters=num_clusters,init='k-means++', max_iter=300, n_init=10, random_state=0)
 kmeans.fit(df)
 cluster_labels = kmeans.labels_
 
 # silhouette score
 silhouette_avg.append(metrics.silhouette_score(df, cluster_labels))

plt.plot(range_n_clusters,silhouette_avg,'bX-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()
#code from https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
```


    
![png](/images/output_85_0.png)
    


According to Silhouette scores, 3 is the best number of clusters

### Getting all plots for 3 clusters


```python
# temp = X.copy()
# model = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
# model.fit(X)
# temp['Labels'] = model.labels_
# sns.pairplot(data=temp,hue='Labels')
```




    <seaborn.axisgrid.PairGrid at 0x20f1b7a08b0>




    
![png](/images/output_88_1.png)
    


From the pair plot we can see that MonthlyRate and Monthly Income has good scatterplot

So, here are some visualizations 


```python
temp = X.copy()
model = KMeans(n_clusters=2,init='k-means++', max_iter=300, n_init=10, random_state=0)
model.fit(X)
y_true3 = model.labels_
temp['Labels'] =model.predict(temp)
ax = sns.scatterplot(data = temp,Y='MonthlyIncome',x= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)
ax = sns.scatterplot(model.cluster_centers_[:, 14], model.cluster_centers_[:, 13],
                     hue=range(2), ec='black', legend=False, ax=ax,palette='magma',s=100)
plt.show()

ax = sns.scatterplot(data = temp,x='TotalWorkingYears',Y= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)
              
plt.show()

ax = sns.scatterplot(data = temp,x='YearsWithCurrManager',Y= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)
              
plt.show()

```

    C:\Python39\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments with/images/out an explicit keyword will result in an error or misinterpretation.
      warnings.warn(



    
![png](/images/output_90_1.png)
    



    
![png](/images/output_90_2.png)
    



    
![png](/images/output_90_3.png)
    


## Metrics to measure K-means
reference: https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6

Metrics Explained:
### 1)The Rand Index
#### The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
### 2)Adjusted Mutual Information (AMI)
#### It is an adjustment of the Mutual Information (MI) score to account for chance. It accounts for the fact that the MI is generally higher for two clusterings with a larger number of clusters, regardless of whether there is actually more information shared. 
### 3)Calinski-Harabasz Index
#### The Calinski-Harabasz index also known as the Variance Ratio Criterion, is the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters, the higher the score , the better the performances.

### 4)Silhouette Coefficient
#### The Silhouette Coefficient is defined for each sample and is composed of two scores(shown in below), and a higher Silhouette Coefficient score relates to a model with better defined clusters.

### 5)Davies-Bouldin Index
#### If the ground truth labels are not known, the Davies-Bouldin index (sklearn.metrics.davies_bouldin_score) can be used to evaluate the model, where a lower Davies-Bouldin index relates to a model with better separation between the clusters. This index signifies the average ‘similarity’ between clusters, where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves.

### Experiments with K-means

I have Made a loop at does clustering from 2 ot 8 clusters and gets the metrics


```python

for i in range(2,9):

    temp = X.copy()
    model = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    model.fit(temp)
    y_pred = model.labels_
    temp['Labels'] = model.predict(temp)
    print("For ",i," clusters, K-means metrics are: ")
    print('Rand score is:', metrics.rand_score(Y, y_pred))
    print('Mutual Info Score is ',metrics.mutual_info_score(Y, y_pred, contingency=None))
    print('calinski harabasz score is ', metrics.calinski_harabasz_score(temp, y_pred))
    print('Silhouette is ', metrics.silhouette_score(temp, y_pred, metric='euclidean'))
    print('Davies Bouldin Score is ', metrics.davies_bouldin_score(X, y_pred))
    print('\n')
    
    
```

    For  2  clusters, K-means metrics are: 
    Rand score is: 0.4997930009308012
    Mutual Info Score is  3.5379315627026564e-05
    calinski harabasz score is  1626.5113320406126
    Silhouette is  0.46621858559531953
    Davies Bouldin Score is  0.8404706489188374
    
    
    For  3  clusters, K-means metrics are: 
    Rand score is: 0.44073945439305745
    Mutual Info Score is  0.005323828302179409
    calinski harabasz score is  1588.6192476956153
    Silhouette is  0.4930177650537684
    Davies Bouldin Score is  0.7965972622636319
    
    
    For  4  clusters, K-means metrics are: 
    Rand score is: 0.39014184298634363
    Mutual Info Score is  0.007389227961190747
    calinski harabasz score is  1589.9984422466014
    Silhouette is  0.4170891352786702
    Davies Bouldin Score is  0.7695961882890432
    
    
    For  5  clusters, K-means metrics are: 
    Rand score is: 0.37112200904868414
    Mutual Info Score is  0.0074267561731437524
    calinski harabasz score is  1757.8865364914102
    Silhouette is  0.4347248078232741
    Davies Bouldin Score is  0.7310995211016834
    
    
    For  6  clusters, K-means metrics are: 
    Rand score is: 0.34981268205035587
    Mutual Info Score is  0.011415953396712957
    calinski harabasz score is  1716.1141422127541
    Silhouette is  0.3872070939383124
    Davies Bouldin Score is  0.8067778044139008
    
    
    For  7  clusters, K-means metrics are: 
    Rand score is: 0.34400929875013314
    Mutual Info Score is  0.009791958374082017
    calinski harabasz score is  1679.9908920939722
    Silhouette is  0.39574122849420634
    Davies Bouldin Score is  0.8110396768134358
    
    
    For  8  clusters, K-means metrics are: 
    Rand score is: 0.3374696100359817
    Mutual Info Score is  0.008406268837334288
    calinski harabasz score is  1710.2230554526232
    Silhouette is  0.4006368474614582
    Davies Bouldin Score is  0.7942352061866016
    
    


### Transformations using PCA


```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca = PCA(3)
pca_data = pca.fit_transform(X)
pca_data.shape
```




    (1470, 3)




```python
model = KMeans(n_clusters=3,init='k-means++', max_iter=300, n_init=10, random_state=0)
model.fit(pca_data)
y_pred = model.labels_
print("For ",3," clusters, K-means metrics are with PCA: ")
print('Rand score is:', metrics.rand_score(Y, y_pred))
print('Mutual Info Score is ',metrics.mutual_info_score(Y, y_pred, contingency=None))
print('calinski harabasz score is ', metrics.calinski_harabasz_score(X, y_pred))
print('Silhouette is ', metrics.silhouette_score(pca_data, y_pred, metric='euclidean'))
print('Davies Bouldin Score is ', metrics.davies_bouldin_score(X, y_pred))
print('\n')
```

    For  3  clusters, K-means metrics are with pca: 
    Rand score is: 0.44073945439305745
    Mutual Info Score is  0.005323828302179409
    calinski harabasz score is  1588.6192324153008
    Silhouette is  0.4930284294904451
    Davies Bouldin Score is  0.7965972622636319
    
    




Different Clustering algorithms 

### Gaussian mixture models (GM)


```python
from sklearn.mixture import GaussianMixture
temp = X.copy()
model = GaussianMixture(n_components=3, random_state=42,covariance_type ='full').fit(temp)
y_true3 = model.predict(temp)
temp['Labels'] = model.predict(temp)


# sns.pairplot(data=temp,hue='Labels')
ax = sns.scatterplot(data = temp,Y='MonthlyIncome',x= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)
                   
plt.show()


print("For 3 clusters, GaussianMixture metrics are: ")
print('Rand score is:', metrics.rand_score(Y, y_true3))
print('Mutual Info Score is ',metrics.mutual_info_score(Y, y_true3, contingency=None))
print('calinski harabasz score is ', metrics.calinski_harabasz_score(X, y_true3))
print('Silhouette is ', metrics.silhouette_score(X, y_true3, metric='euclidean'))
print('Davies Bouldin Score is ', metrics.davies_bouldin_score(X, y_true3))
```


    
![png](/images/output_100_0.png)
    


    For 3 clusters, GaussianMixture metrics are: 
    Rand score is: 0.4251520077057372
    Mutual Info Score is  0.018547497512441113
    calinski harabasz score is  131.63262344350454
    Silhouette is  0.01855533334536094
    Davies Bouldin Score is  11.952181025378104


### Hierarchical Clustering using Agglomerative Clustering


```python
from sklearn.cluster import AgglomerativeClustering

temp = X.copy()
model = AgglomerativeClustering(n_clusters=3).fit(temp)
y_true3 = model.labels_
temp['Labels'] =  model.labels_


# sns.pairplot(data=temp,hue='Labels')
ax = sns.scatterplot(data = temp,Y='MonthlyIncome',x= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)
                   
plt.show()


print("For 3 clusters, Agglomerative Clustering metrics are: ")
print('Rand score is:', metrics.rand_score(Y, y_true3))
print('Mutual Info Score is ',metrics.mutual_info_score(Y, y_true3, contingency=None))
print('calinski harabasz score is ', metrics.calinski_harabasz_score(X, y_true3))
print('Silhouette is ', metrics.silhouette_score(X, y_true3, metric='euclidean'))
print('Davies Bouldin Score is ', metrics.davies_bouldin_score(X, y_true3))

```


    
![png](/images/output_102_0.png)
    


    For 3 clusters, GaussianMixture metrics are: 
    Rand score is: 0.4419027243300315
    Mutual Info Score is  0.004803069051401772
    calinski harabasz score is  1245.3170041273338
    Silhouette is  0.41090328152908645
    Davies Bouldin Score is  0.9111318344593142


### Hierarchical Clustering Dendrogram
code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html


```python
from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)




# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
```


    
![png](/images/output_104_0.png)
    


### MiniBatch


```python
temp = X.copy()
model = MiniBatchKMeans(n_clusters=3,
                          random_state=42,
                          batch_size=6,
                          max_iter=10).fit(temp)

temp['Labels'] =model.predict(temp)
y_true3=model.labels_

ax = sns.scatterplot(data = temp,Y='MonthlyIncome',x= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)

plt.show()


print("For 3 clusters, MiniBatchKMeans metrics are: ")
print('Rand score is:', metrics.rand_score(Y, y_true3))
print('Mutual Info Score is ',metrics.mutual_info_score(Y, y_true3, contingency=None))
print('calinski harabasz score is ', metrics.calinski_harabasz_score(X, y_true3))
print('Silhouette is ', metrics.silhouette_score(X, y_true3, metric='euclidean'))
print('Davies Bouldin Score is ', metrics.davies_bouldin_score(X, y_true3))
```


    
![png](/images/output_106_0.png)
    


    For 3 clusters, MiniBatchKMeans metrics are: 
    Rand score is: 0.4390241869382198
    Mutual Info Score is  0.003888247780669693
    calinski harabasz score is  1474.1770849296245
    Silhouette is  0.46474154024296294
    Davies Bouldin Score is  0.8585046064093943


## Affinity Propagation


```python
temp = X.copy()
model =  AffinityPropagation( random_state=0).fit(temp)

temp['Labels'] =model.predict(temp)
y_true3=model.labels_

ax = sns.scatterplot(data = temp,Y='MonthlyIncome',x= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)

plt.show()


print("AffinityPropagation metrics are: ")
print('Rand score is:', metrics.rand_score(Y, y_true3))
print('Mutual Info Score is ',metrics.mutual_info_score(Y, y_true3, contingency=None))
print('calinski harabasz score is ', metrics.calinski_harabasz_score(X, y_true3))
print('Silhouette is ', metrics.silhouette_score(X, y_true3, metric='euclidean'))
print('Davies Bouldin Score is ', metrics.davies_bouldin_score(X, y_true3))
```


    
![png](/images/output_108_0.png)
    


    MiniBatchKMeans metrics are: 
    Rand score is: 0.28729896315231335
    Mutual Info Score is  0.032638360796187066
    calinski harabasz score is  1660.7874821332443
    Silhouette is  0.3373943983241517
    Davies Bouldin Score is  0.8636222911854409


## Mean Shift


```python
from sklearn.cluster import MeanShift,estimate_bandwidth
temp = X.copy()
model =   MeanShift(n_jobs=-1).fit(temp)

temp['Labels'] =model.predict(temp)
y_true3=model.labels_

ax = sns.scatterplot(data = temp,Y='MonthlyIncome',x= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)

plt.show()


print("MeanShift metrics are: ")
print('Rand score is:', metrics.rand_score(Y, y_true3))
print('Mutual Info Score is ',metrics.mutual_info_score(Y, y_true3, contingency=None))
print('calinski harabasz score is ', metrics.calinski_harabasz_score(X, y_true3))
print('Silhouette is ', metrics.silhouette_score(X, y_true3, metric='euclidean'))
print('Davies Bouldin Score is ', metrics.davies_bouldin_score(X, y_true3))
```


    
![png](/images/output_110_0.png)
    


    MiniBatchKMeans metrics are: 
    Rand score is: 0.5061946902654867
    Mutual Info Score is  3.9152102377149056e-05
    calinski harabasz score is  1530.0493674527152
    Silhouette is  0.45252411040016804
    Davies Bouldin Score is  0.8428060105585939


## Spectral Clustering


```python
from sklearn.cluster import  SpectralClustering
temp = X.copy()
model =   SpectralClustering(n_clusters=3,  random_state=42).fit(temp)

temp['Labels'] =model.labels_
y_true3=model.labels_

ax = sns.scatterplot(data = temp,Y='MonthlyIncome',x= 'MonthlyRate', hue='Labels', palette="Set2", alpha=0.7)

plt.show()


print("Spectral Clustering metrics are: ")
print('Rand score is:', metrics.rand_score(Y, y_true3))
print('Mutual Info Score is ',metrics.mutual_info_score(Y, y_true3, contingency=None))
print('calinski harabasz score is ', metrics.calinski_harabasz_score(X, y_true3))
print('Silhouette is ', metrics.silhouette_score(X, y_true3, metric='euclidean'))
print('Davies Bouldin Score is ', metrics.davies_bouldin_score(X, y_true3))
```

    C:\Python39\lib\site-packages\sklearn\manifold\_spectral_embedding.py:260: UserWarning: Graph is not fully connected, spectral embedding may not work as expected.
      warnings.warn(



    
![png](/images/output_112_1.png)
    


    Spectral Clustering metrics are: 
    Rand score is: 0.696295781757223
    Mutual Info Score is  0.0020836176615380703
    calinski harabasz score is  1.8440776520266107
    Silhouette is  -0.13354008727624947
    Davies Bouldin Score is  7.041660915699087


# Conclusion

From all the different types of clustering algorithms I have used, Spectral Clustering gives me the best Rand score of 0.69 but negative Silhouette score. Affinity propagation gave the lowest rand score of 0.28 but with a Silhouette score of 0.33.
Spectral Clustering gave a rand score more than K-Means which had a high score of 0.49

Comparing clustering to classification done in part-1, the classification performed better than clustering as it had better accuracy of 78% compared to 0.49 for kmeans or even 0.69 for Spectral Clustering.
And with feature selection the accuracy if classification went uo to 87%.

But if the data was unlabeled, then it is impressive how high the accuracy is for clustering given no context is there while clustering. 



# Supervised Learning: Generalisation & Overfitting

# Decision trees

DecisionTreeClassifier


```python
from sklearn import tree
from sklearn.model_selection import cross_val_score

clf = tree.DecisionTreeClassifier(random_state=200)
```

## Using 10 fold cross validation 


```python
from sklearn.model_selection import cross_val_score
scoring = ['accuracy','precision','recall','f1','roc_auc']
for i in scoring:
    scores = cross_val_score(clf, X, Y, cv=10,n_jobs=-1,scoring=i)
    print("Mean",i,"of decsion tree is ",scores.mean())


```

    Mean accuracy of decsion tree is  0.7904761904761906
    Mean precision of decsion tree is  0.36008831724610124
    Mean recall of decsion tree is  0.4003623188405797
    Mean f1 of decsion tree is  0.37696254959002296
    Mean roc_auc of decsion tree is  0.6328555627772268



```python
from sklearn.tree import plot_tree
clf.fit(X,Y)
plot_tree(clf);

```


    
![png](/images/output_123_0.png)
    



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=0)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
```


```python
from pycm import ConfusionMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix,roc_auc_score,plot_roc_curve
cm = ConfusionMatrix(actual_vector=np.array(y_test), predict_vector=np.array(y_pred))

TPR = cm.class_stat.get("TPR")
FPR = cm.class_stat.get("FPR")
TNR = cm.class_stat.get("TNR")
FNR  = cm.class_stat.get("FNR")
ACC = cm.class_stat.get("ACC")
F1 = cm.class_stat.get("F1")
AUC= cm.class_stat.get("AUC")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_area = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
```


```python
print("Accuracy : ", ACC)
print("Precision : ", precision)
print("Recall : ", recall)
print("True Positive Rate : ", TPR)
print("False Positive Rate : ", FPR)
print("True Negative Rate : ", TNR)
print("False Negative Rate : ", FNR)
print("F1 Score : ", F1)
print("ROC Area : ", AUC)
```

    Accuracy :  {0: 0.8179347826086957, 1: 0.8179347826086957}
    Precision :  0.4
    Recall :  0.3103448275862069
    True Positive Rate :  {0: 0.9129032258064517, 1: 0.3103448275862069}
    False Positive Rate :  {0: 0.6896551724137931, 1: 0.08709677419354833}
    True Negative Rate :  {0: 0.3103448275862069, 1: 0.9129032258064517}
    False Negative Rate :  {0: 0.08709677419354833, 1: 0.6896551724137931}
    F1 Score :  {0: 0.8941548183254344, 1: 0.34951456310679613}
    ROC Area :  {0: 0.6116240266963293, 1: 0.6116240266963293}


Conclusion : Accuracy while doing one test train split is higher (81%) than the mean accuracy (79&) of doing a 10 fold cross validation 

## Experimenting with Decision Tree parameters 


```python
def Test_Tree_1(X_1,Y_1,max_features1,max_depth1,splitter,min_samples_split):
    clf = tree.DecisionTreeClassifier(max_features= max_features1,max_depth= max_depth1,splitter= splitter,min_samples_split= min_samples_split,random_state=200)
    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1,random_state=0)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    cm = ConfusionMatrix(actual_vector=np.array(y_test), predict_vector=np.array(y_pred))
    ACC = cm.class_stat.get("ACC")
    return ACC
```

## Effect of minimum samples per leaf 


```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
test_scores,para_value,depth_value,leaf_number = list(), list(),list(),list()


for i in range(1, 100):
	# configure the model
	model = DecisionTreeClassifier(min_samples_leaf=i)
	para_value.append(i)
	model.fit(X_train, y_train)
	depth_value.append(model.get_depth())
	leaf_number.append(model.get_n_leaves())
	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth

fig, axs = plt.subplots(4, 1,figsize=(13,13))
axs[0].plot(para_value, test_scores, '-o', label='min_samples vs accuracy')
axs[0].set_title('min_samples vs accuracy')
axs[1].plot(para_value, depth_value, '-o', label='min_samples vs tree size')
axs[1].set_title('min_samples vs tree size')
axs[2].plot( depth_value,test_scores, '-o', label='tree size vs accuracy')
axs[2].set_title('tree size vs accuracy')
axs[3].plot( leaf_number,test_scores, '-o', label='number of leaves vs accuracy')
axs[3].set_title('number of leaves vs accuracy')


axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()
```


    
![png](/images/output_131_0.png)
    


Conclusion:Accuracy increases as min number of samples per leaf increases.
As the number of samples per leaf increases, the tree size decreases.

## Effect of max features on tree depth 


```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
test_scores,para_value,depth_value = list(), list(),list()


for i in range(1, 27):
	# configure the model
	model = DecisionTreeClassifier(max_features=i)
	para_value.append(i)
	model.fit(X_train, y_train)
	depth_value.append(model.get_depth())
	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth

fig, axs = plt.subplots(2, 1,figsize=(7,7))
axs[0].plot(para_value, test_scores, '-o', label='max_features vs accuracy')
axs[0].set_title('max_features vs accuracy')
axs[1].plot(para_value, depth_value, '-o', label='max_features vs tree size')
axs[1].set_title('max_features vs tree size')



axs[0].legend()
axs[1].legend()

plt.show()
```


    
![png](/images/output_134_0.png)
    


Conclusion: As max number of features used increases, tree size decareses. Also the accuracy increases as more features are used.  

## Effect of max depth on accuracy


```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
test_scores,para_value,depth_value = list(), list(),list()


for i in range(1, 100):
	# configure the model
	model = DecisionTreeClassifier(max_depth=i)
	para_value.append(i)
	model.fit(X_train, y_train)
	depth_value.append(model.get_depth())
	test_yhat = model.predict(X_test)
	test_acc = metrics.accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth

fig, axs = plt.subplots(2, 1,figsize=(14,14))
axs[0].plot(para_value, test_scores, '-o', label='max_depth vs accuracy')
axs[0].set_title('max_depth vs accuracy')
axs[1].plot(para_value, depth_value, '-o', label='max_depth vs tree size')
axs[1].set_title('max_depth vs tree size')




axs[0].legend()
axs[1].legend()


plt.show()
```


    
![png](/images/output_137_0.png)
    


Conclusion: As the depth of the tree increases, accuracy drops initially and then it becomes constant ranging from 83% to 80%. In second plot, the maximum depth reached saturates at 16.

## Effect of splitting criteria on tree size


```python
model = DecisionTreeClassifier(splitter='best')
model.fit(X_train, y_train)
test_yhat = model.predict(X_test)
print('splitting criteria : "best" gives max depth of ',model.get_depth(), 'and an accuracy of ',accuracy_score(y_test, test_yhat) )

model = DecisionTreeClassifier(splitter='random')
model.fit(X_train, y_train)
test_yhat = model.predict(X_test)
print('splitting criteria : "random" gives max depth of ',model.get_depth(), 'and an accuracy of ',accuracy_score(y_test, test_yhat) )

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
test_yhat = model.predict(X_test)
print('criterion : "entropy" gives max depth of ',model.get_depth(), 'and an accuracy of ',accuracy_score(y_test, test_yhat) )

model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)
test_yhat = model.predict(X_test)
print('criterion : "gini" gives max depth of ',model.get_depth(), 'and an accuracy of ',accuracy_score(y_test, test_yhat) )



```

    splitting criteria : "best" gives max depth of  15 and an accuracy of  0.8152173913043478
    splitting criteria : "random" gives max depth of  16 and an accuracy of  0.7989130434782609
    criterion : "entropy" gives max depth of  13 and an accuracy of  0.7581521739130435
    criterion : "gini" gives max depth of  15 and an accuracy of  0.8125


## Using Random Search to find optimal hyper parameters 


```python
from random import randint
max_feature_list_best = 0
max_depth_best = 0 
splitter_best = 0 
splitter_select_best = 0 
mini_samples_best  = 0
best_accuracy = 0
splitter = ['best','random']

for i in range(1,1000):
    max_feature_list =  randint(5, 27)
    max_depth = randint(2, 100)
    splitter_select = splitter[randint(0,1)]
    mini_samples = randint(2, 100)
    temp = Test_Tree_1(X,Y,max_feature_list,max_depth,splitter_select,mini_samples)[0]
    if (temp > best_accuracy):
        best_accuracy = temp
        max_feature_list_best = max_feature_list
        max_depth_best = clf.get_depth()
        splitter_best = splitter_select
        mini_samples_best = mini_samples

print("Highest accuracy achieved is ",best_accuracy) 
print('With ',max_feature_list_best, ' number of features' )
print('With ',max_depth_best, ' maximum depth' )
print('With ',splitter_best, 'as splitting criteria ' )
print('With ',mini_samples_best, ' as mini number of samples ' )



```

    Highest accuracy achieved is  0.8804347826086957
    With  20  number of features
    With  12  maximum depth
    With  random as splitting criteria 
    With  27  as mini number of samples 



```python
def Test_Tree(X_1,Y_1,split_train):
# ,max_features_1,max_depth1,splitter,min_samples_split,split_train):
    
    clf = tree.DecisionTreeClassifier(random_state=200)
    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1,random_state=0,test_size=split_train)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    cm = ConfusionMatrix(actual_vector=np.array(y_test), predict_vector=np.array(y_pred))
    TPR = cm.class_stat.get("TPR")
    FPR = cm.class_stat.get("FPR")
    TNR = cm.class_stat.get("TNR")
    FNR  = cm.class_stat.get("FNR")
    ACC = cm.class_stat.get("ACC")
    F1 = cm.class_stat.get("F1")
    AUC= cm.class_stat.get("AUC")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_area = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

    print("Accuracy : ", ACC)
    print("Precision : ", precision)
    print("Recall : ", recall)
    print("True Positive Rate : ", TPR)
    print("False Positive Rate : ", FPR)
    print("True Negative Rate : ", TNR)
    print("False Negative Rate : ", FNR)
    print("F1 Score : ", F1)
    print("ROC Area : ", AUC)
    
```

## Accuracy with 30% of data as test set


```python
Test_Tree(X,Y, 0.3)
```

    Accuracy :  {0: 0.780045351473923, 1: 0.780045351473923}
    Precision :  0.2857142857142857
    Recall :  0.2571428571428571
    True Positive Rate :  {0: 0.8787061994609164, 1: 0.2571428571428571}
    False Positive Rate :  {0: 0.7428571428571429, 1: 0.12129380053908356}
    True Negative Rate :  {0: 0.2571428571428571, 1: 0.8787061994609164}
    False Negative Rate :  {0: 0.12129380053908356, 1: 0.7428571428571429}
    F1 Score :  {0: 0.8704939919893191, 1: 0.2706766917293233}
    ROC Area :  {0: 0.5679245283018868, 1: 0.5679245283018868}


## Accuracy with 60% of data as test set


```python
Test_Tree(X,Y, 0.6)
```

    Accuracy :  {0: 0.7562358276643991, 1: 0.7562358276643991}
    Precision :  0.287292817679558
    Recall :  0.37681159420289856
    True Positive Rate :  {0: 0.8266129032258065, 1: 0.37681159420289856}
    False Positive Rate :  {0: 0.6231884057971014, 1: 0.1733870967741935}
    True Negative Rate :  {0: 0.37681159420289856, 1: 0.8266129032258065}
    False Negative Rate :  {0: 0.1733870967741935, 1: 0.6231884057971014}
    F1 Score :  {0: 0.8512110726643599, 1: 0.32601880877742945}
    ROC Area :  {0: 0.6017122487143525, 1: 0.6017122487143525}


### Conclusion: It can be seen that when using 30% of data for testing, it is over-fitting as it has more accuracy of 78.00% compared to using 60% of data for testing with accuracy of 75.62%

## Visualization of over-fitting using accuracy


```python
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 50)]
# evaluate a decision tree for each depth
for i in values:
	# configure the model
	model = DecisionTreeClassifier(max_depth=i)
	# fit model on the training dataset
	model.fit(X_train, y_train)
	# evaluate on the train dataset
	train_yhat = model.predict(X_train)
	train_acc = accuracy_score(y_train, train_yhat)
	train_scores.append(train_acc)
	# evaluate on the test dataset
	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()
```


    
![png](/images/output_150_0.png)
    


Conclusion: It can be observed that it starts to over-fit once the depth reaches 13 and more. Thus having more depth leads to over-fitting.

# Using different decision tree algorithms 

## Random Forest 


```python
from sklearn.ensemble import RandomForestClassifier
def Test_Random_Tree(X_1,Y_1,split =0.5):
    
    clf = RandomForestClassifier(random_state=0,n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1,random_state=0,test_size=split)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    cm = ConfusionMatrix(actual_vector=np.array(y_test), predict_vector=np.array(y_pred))
    TPR = cm.class_stat.get("TPR")
    FPR = cm.class_stat.get("FPR")
    TNR = cm.class_stat.get("TNR")
    FNR  = cm.class_stat.get("FNR")
    ACC = cm.class_stat.get("ACC")
    F1 = cm.class_stat.get("F1")
    AUC= cm.class_stat.get("AUC")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_area = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

    print("Accuracy : ", ACC)
    print("Precision : ", precision)
    print("Recall : ", recall)
    print("True Positive Rate : ", TPR)
    print("False Positive Rate : ", FPR)
    print("True Negative Rate : ", TNR)
    print("False Negative Rate : ", FNR)
    print("F1 Score : ", F1)
    print("ROC Area : ", AUC)

```


```python
Test_Random_Tree(X,Y)
```

    Accuracy :  {0: 0.8721088435374149, 1: 0.8721088435374149}
    Precision :  0.8461538461538461
    Recall :  0.19642857142857142
    True Positive Rate :  {0: 0.9935794542536116, 1: 0.19642857142857142}
    False Positive Rate :  {0: 0.8035714285714286, 1: 0.006420545746388395}
    True Negative Rate :  {0: 0.19642857142857142, 1: 0.9935794542536116}
    False Negative Rate :  {0: 0.006420545746388395, 1: 0.8035714285714286}
    F1 Score :  {0: 0.9294294294294294, 1: 0.3188405797101449}
    ROC Area :  {0: 0.5950040128410915, 1: 0.5950040128410915}


## Testing Random Forest for over-fitting 

### For 30% test size


```python
Test_Random_Tree(X,Y,0.3)
```

    Accuracy :  {0: 0.8639455782312925, 1: 0.8639455782312925}
    Precision :  0.8125
    Recall :  0.18571428571428572
    True Positive Rate :  {0: 0.9919137466307277, 1: 0.18571428571428572}
    False Positive Rate :  {0: 0.8142857142857143, 1: 0.008086253369272267}
    True Negative Rate :  {0: 0.18571428571428572, 1: 0.9919137466307277}
    False Negative Rate :  {0: 0.008086253369272267, 1: 0.8142857142857143}
    F1 Score :  {0: 0.9246231155778895, 1: 0.3023255813953488}
    ROC Area :  {0: 0.5888140161725067, 1: 0.5888140161725067}


### For 60% test size


```python
Test_Random_Tree(X,Y,0.6)

```

    Accuracy :  {0: 0.8582766439909297, 1: 0.8582766439909297}
    Precision :  0.7096774193548387
    Recall :  0.15942028985507245
    True Positive Rate :  {0: 0.9879032258064516, 1: 0.15942028985507245}
    False Positive Rate :  {0: 0.8405797101449275, 1: 0.012096774193548376}
    True Negative Rate :  {0: 0.15942028985507245, 1: 0.9879032258064516}
    False Negative Rate :  {0: 0.012096774193548376, 1: 0.8405797101449275}
    F1 Score :  {0: 0.9216300940438872, 1: 0.2603550295857988}
    ROC Area :  {0: 0.573661757830762, 1: 0.573661757830762}


Conclusion:  The difference between the accuracies of 30% and 60% test size is less than that of decision trees.  
This is because random forrest groups the results of many decesion trees which deceases the effect of over-fitting. 


```python
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

train_scores, test_scores = list(), list()
# define the tree depths to evaluate
values = [i for i in range(1, 50)]
# evaluate a decision tree for each depth
for i in values:
	# configure the model
	model = RandomForestClassifier(max_depth=i)
	# fit model on the training dataset
	model.fit(X_train, y_train)
	# evaluate on the train dataset
	train_yhat = model.predict(X_train)
	train_acc = accuracy_score(y_train, train_yhat)
	train_scores.append(train_acc)
	# evaluate on the test dataset
	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()
```


    
![png](/images/output_162_0.png)
    



```python
def Test_DecisionTreeRegressor(X_1,Y_1,split=0.5):
    
    clf = tree.DecisionTreeRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1,random_state=0,test_size=split)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    cm = ConfusionMatrix(actual_vector=np.array(y_test), predict_vector=np.array(y_pred))

    ACC = cm.class_stat.get("ACC")
 

    print("Accuracy : ", ACC)
    print("Precision : ", precision)
    print("Recall : ", recall)


```


```python
Test_DecisionTreeRegressor(X,Y)
```

    Accuracy :  {'0': 0.1523809523809524, '0.0': 0.1673469387755102, '1': 0.8476190476190476, '1.0': 0.8326530612244898}
    Precision :  0.4
    Recall :  0.3103448275862069


Final Conclusion:

Over-Fitting of Decision trees: Using 10 fod cross validation leads to a lower average score then running the decesion tree classifier once. 
Also using more test data split, leads to lower accuracy than with less test data. Both the above phenomenons are due to over-fitting. Using More training data overf-its the tree to the dataset and thus leads to higher accuracy. 

The experiments with parameters showed the greater tree sizes leads to lower accuracy and other conclusions related to parameters are mentioned below each experiment.

Using different Algorithm: Decision trees are prone to over-fitting as seen in the experiments, having more training data (less test data size) has higher accuracy difference than having less training data (more test data size). 
Random forest on the other hand is not as prone to over-fitting as Decision trees as it produces a list of Decision trees and gets the aggregate of all the tress.

# Stage - 4: Neural Network

## Stochastic Gradient Descent (SGD)


```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=0)

clf = SGDClassifier( max_iter=5,n_jobs=-1)

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
#roc_area = metrics.roc_auc_score(y_test, SGDClassifier.predict_proba(X_test)[:,1])
print("With/images/out 10-fold cross validation:")
print("accuracy : ", accuracy)

print("Using 10-fold cross validation:")
scoring = ['accuracy']
for i in scoring:
    scores = cross_val_score(clf, X, Y, cv=10,n_jobs=-1,scoring=i)
    print(i,": ",scores.mean())
```

    With/images/out 10-fold cross validation:
    accuracy :  0.842391304347826
    Using 10-fold cross validation:
    accuracy :  0.7646258503401361


### Conclusion: The Linear classifier generalizes well with the data. This is shown by the high accuracy of 84.2%. But using 10 fold cross validation, the accuracy drops off to 70.3%.  The accuracy is still high which means the dataset is linearly sparable. 

## MLP Classifier 


```python
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(27,270,150,2) , activation='relu' ,random_state=1, max_iter=2, learning_rate='adaptive').fit(X_train, y_train)

y_pred=clf.predict(X_train)
accuracy = metrics.accuracy_score(y_train, y_pred)
print("Accuracy with y_train : ", accuracy)

y_pred=clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy with y_test : ", accuracy)


```

    Accuracy with y_train :  0.837568058076225
    Accuracy with y_test :  0.842391304347826


    C:\Python39\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:692: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (2) reached and the optimization hasn't converged yet.
      warnings.warn(


## Experiment with Activation functions 


```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

test_scores=[]
para_value=[]


act_fun= ['identity','logistic','tanh','relu']
counter= 0
for i in act_fun:
	counter+=1
	# configure the model
	model = MLPClassifier(hidden_layer_sizes=(27,270,150,2) ,activation=i)
	para_value.append(counter)
	model.fit(X_train, y_train)
	test_yhat = model.predict(X_test)
	test_acc = metrics.accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth

import matplotlib.pyplot as plt

plt.bar(act_fun,test_scores,align='center') # A bar chart
plt.xlabel('Activation')
plt.ylabel('Accuracy')
 # Here you are drawing the horizontal lines
plt.show()







plt.show()
```


    
![png](/images/output_173_0.png)
    


## Experimenting with size of layers


```python
def convert(list):
    return tuple(list)
```


```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

test_scores=[]
para_value=[]


layers = []
for i in range(10):
	layers.append(50)
	layer = convert(layers)
	# configure the model
	model = MLPClassifier(hidden_layer_sizes=(layer))
	para_value.append(i)
	model.fit(X_train, y_train)
	test_yhat = model.predict(X_test)
	test_acc = metrics.accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth

import matplotlib.pyplot as plt

plt.plot(para_value,test_scores) # A bar chart
plt.xlabel('Hidden Layer size')
plt.ylabel('Accuracy')
 # Here you are drawing the horizontal lines
plt.show()


```


    
![png](/images/output_176_0.png)
    


## Experiment with differnt learning rate  


```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
test_scores=[]
para_value=[]
depth_value = []
counter = 0

lr= [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
for i in lr:
	counter+=1
	# configure the model
	model = MLPClassifier(learning_rate_init=i)
	para_value.append(i)
	model.fit(X_train, y_train)
	test_yhat = model.predict(X_test)
	test_acc = metrics.accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth

import matplotlib.pyplot as plt

plt.plot(para_value,test_scores) # A bar chart
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
 # Here you are drawing the horizontal lines
plt.show()



```


    
![png](/images/output_178_0.png)
    


## Experiment with momentum 


```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
test_scores,para_value,depth_value = list(), list(),list()

lr= [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
for i in lr:
	counter+=1
	# configure the model
	model = MLPClassifier(solver = 'sgd',momentum=i)
	para_value.append(i)
	model.fit(X_train, y_train)
	test_yhat = model.predict(X_test)
	test_acc = metrics.accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth

import matplotlib.pyplot as plt

plt.plot(para_value,test_scores) # A bar chart
plt.xlabel('Momentum')
plt.ylabel('Accuracy')

plt.show()



```


    
![png](/images/output_180_0.png)
    


## Experiment with validation  threshold 


```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
test_scores,para_value,depth_value = list(), list(),list()


for i in range(1,20):
	counter+=1
	# configure the model
	model = MLPClassifier(solver = 'sgd', n_iter_no_change=i)
	para_value.append(i)
	model.fit(X_train, y_train)
	test_yhat = model.predict(X_test)
	test_acc = metrics.accuracy_score(y_test, test_yhat)
	test_scores.append(test_acc)
# plot of train and test scores vs tree depth

import matplotlib.pyplot as plt

plt.plot(para_value,test_scores) # A bar chart
plt.xlabel('validation  threshold')
plt.ylabel('Accuracy')

plt.show()


```


    
![png](/images/output_182_0.png)
    


Conclusion: Conclusion: There was no change observed in changing any of the hyperparameters like layer depth, activations functions, momentum, learning rate etc. Other than learning rate getting an accuracy of 84.5%, all of them gave 84.2% as the accuracy. 

# Using Dense layers from Keras 


```python
import tensorflow as tf
print(tf.__version__)
```

    2.6.0



```python

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(27,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x=X_train,y=y_train,epochs=6000)
```

    Epoch 1/6000
    35/35 [==============================] - 0s 471us/step - loss: 821.9313 - accuracy: 0.4510
    Epoch 2/6000
    35/35 [==============================] - 0s 441us/step - loss: 20.8266 - accuracy: 0.7713
    Epoch 3/6000
    35/35 [==============================] - 0s 441us/step - loss: 11.1189 - accuracy: 0.7450
    Epoch 4/6000
    35/35 [==============================] - 0s 471us/step - loss: 15.7991 - accuracy: 0.7296
    Epoch 5/6000
    35/35 [==============================] - 0s 471us/step - loss: 14.1944 - accuracy: 0.7477
    Epoch 6/6000
    35/35 [==============================] - 0s 441us/step - loss: 19.7651 - accuracy: 0.7142
    Epoch 7/6000
    35/35 [==============================] - 0s 441us/step - loss: 20.3979 - accuracy: 0.7269
    Epoch 8/6000
    35/35 [==============================] - 0s 471us/step - loss: 12.9815 - accuracy: 0.7459
    Epoch 9/6000
    35/35 [==============================] - 0s 441us/step - loss: 14.0282 - accuracy: 0.7232
    Epoch 10/6000
    35/35 [==============================] - 0s 441us/step - loss: 17.5957 - accuracy: 0.7323
    Epoch 11/6000
    35/35 [==============================] - 0s 441us/step - loss: 16.7081 - accuracy: 0.7423
    Epoch 12/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.1240 - accuracy: 0.7468
    Epoch 13/6000
    35/35 [==============================] - 0s 441us/step - loss: 12.0271 - accuracy: 0.7250
    Epoch 14/6000
    35/35 [==============================] - 0s 441us/step - loss: 12.1431 - accuracy: 0.7287
    Epoch 15/6000
    35/35 [==============================] - 0s 441us/step - loss: 19.3104 - accuracy: 0.7396
    Epoch 16/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.5132 - accuracy: 0.7241
    Epoch 17/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.7007 - accuracy: 0.7486
    Epoch 18/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.2089 - accuracy: 0.7577
    Epoch 19/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.8508 - accuracy: 0.7396
    Epoch 20/6000
    35/35 [==============================] - 0s 471us/step - loss: 18.6245 - accuracy: 0.7377
    Epoch 21/6000
    35/35 [==============================] - 0s 441us/step - loss: 13.9130 - accuracy: 0.7151
    Epoch 22/6000
    35/35 [==============================] - 0s 412us/step - loss: 15.3897 - accuracy: 0.7704
    Epoch 23/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.6104 - accuracy: 0.7405
    Epoch 24/6000
    35/35 [==============================] - 0s 471us/step - loss: 16.1613 - accuracy: 0.7323
    Epoch 25/6000
    35/35 [==============================] - 0s 441us/step - loss: 13.4604 - accuracy: 0.7387
    Epoch 26/6000
    35/35 [==============================] - 0s 441us/step - loss: 13.5952 - accuracy: 0.7459
    Epoch 27/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.8981 - accuracy: 0.7505
    Epoch 28/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.6675 - accuracy: 0.7260
    Epoch 29/6000
    35/35 [==============================] - 0s 441us/step - loss: 28.2082 - accuracy: 0.7495
    Epoch 30/6000
    35/35 [==============================] - 0s 647us/step - loss: 16.8409 - accuracy: 0.7087
    Epoch 31/6000
    35/35 [==============================] - 0s 441us/step - loss: 14.7674 - accuracy: 0.7641
    Epoch 32/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.9085 - accuracy: 0.7332
    Epoch 33/6000
    35/35 [==============================] - 0s 471us/step - loss: 18.4025 - accuracy: 0.7278
    Epoch 34/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.1403 - accuracy: 0.7632
    Epoch 35/6000
    35/35 [==============================] - 0s 441us/step - loss: 17.3745 - accuracy: 0.7323
    Epoch 36/6000
    35/35 [==============================] - 0s 500us/step - loss: 24.2401 - accuracy: 0.7214
    Epoch 37/6000
    35/35 [==============================] - 0s 471us/step - loss: 13.8084 - accuracy: 0.7686
    Epoch 38/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.7364 - accuracy: 0.7432
    Epoch 39/6000
    35/35 [==============================] - 0s 500us/step - loss: 5.0495 - accuracy: 0.7414
    Epoch 40/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.5426 - accuracy: 0.7595
    Epoch 41/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.6041 - accuracy: 0.7359
    Epoch 42/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.8120 - accuracy: 0.7641
    Epoch 43/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.5134 - accuracy: 0.7287
    Epoch 44/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.1304 - accuracy: 0.7541
    Epoch 45/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.1817 - accuracy: 0.7396
    Epoch 46/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.4641 - accuracy: 0.7632
    Epoch 47/6000
    35/35 [==============================] - 0s 471us/step - loss: 12.2781 - accuracy: 0.7260
    Epoch 48/6000
    35/35 [==============================] - 0s 471us/step - loss: 23.3261 - accuracy: 0.7659
    Epoch 49/6000
    35/35 [==============================] - 0s 471us/step - loss: 16.9098 - accuracy: 0.7169
    Epoch 50/6000
    35/35 [==============================] - 0s 471us/step - loss: 10.8079 - accuracy: 0.7704
    Epoch 51/6000
    35/35 [==============================] - 0s 471us/step - loss: 12.6766 - accuracy: 0.7450
    Epoch 52/6000
    35/35 [==============================] - 0s 441us/step - loss: 27.3839 - accuracy: 0.7260
    Epoch 53/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.8320 - accuracy: 0.7441
    Epoch 54/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9715 - accuracy: 0.7722
    Epoch 55/6000
    35/35 [==============================] - 0s 471us/step - loss: 8.2034 - accuracy: 0.7350
    Epoch 56/6000
    35/35 [==============================] - 0s 471us/step - loss: 22.9426 - accuracy: 0.7196
    Epoch 57/6000
    35/35 [==============================] - 0s 471us/step - loss: 12.7392 - accuracy: 0.7623
    Epoch 58/6000
    35/35 [==============================] - 0s 441us/step - loss: 11.2198 - accuracy: 0.7414
    Epoch 59/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9364 - accuracy: 0.7704
    Epoch 60/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9712 - accuracy: 0.7659
    Epoch 61/6000
    35/35 [==============================] - 0s 471us/step - loss: 8.8584 - accuracy: 0.7695
    Epoch 62/6000
    35/35 [==============================] - 0s 441us/step - loss: 16.0573 - accuracy: 0.7205
    Epoch 63/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.4714 - accuracy: 0.7595
    Epoch 64/6000
    35/35 [==============================] - 0s 500us/step - loss: 8.7455 - accuracy: 0.7459
    Epoch 65/6000
    35/35 [==============================] - 0s 471us/step - loss: 14.1988 - accuracy: 0.7142
    Epoch 66/6000
    35/35 [==============================] - 0s 471us/step - loss: 11.6114 - accuracy: 0.7659
    Epoch 67/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.4391 - accuracy: 0.7632
    Epoch 68/6000
    35/35 [==============================] - 0s 500us/step - loss: 16.2002 - accuracy: 0.7568
    Epoch 69/6000
    35/35 [==============================] - 0s 559us/step - loss: 6.1745 - accuracy: 0.7541
    Epoch 70/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.9518 - accuracy: 0.7641
    Epoch 71/6000
    35/35 [==============================] - 0s 471us/step - loss: 10.5609 - accuracy: 0.7423
    Epoch 72/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.9065 - accuracy: 0.7568
    Epoch 73/6000
    35/35 [==============================] - 0s 559us/step - loss: 15.6814 - accuracy: 0.7432
    Epoch 74/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.3379 - accuracy: 0.7387
    Epoch 75/6000
    35/35 [==============================] - 0s 471us/step - loss: 20.4070 - accuracy: 0.7160
    Epoch 76/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.3995 - accuracy: 0.7604
    Epoch 77/6000
    35/35 [==============================] - 0s 500us/step - loss: 5.6765 - accuracy: 0.7568
    Epoch 78/6000
    35/35 [==============================] - 0s 500us/step - loss: 8.2161 - accuracy: 0.7405
    Epoch 79/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.0253 - accuracy: 0.7786
    Epoch 80/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.3957 - accuracy: 0.7750
    Epoch 81/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.4359 - accuracy: 0.7559
    Epoch 82/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.8742 - accuracy: 0.7695
    Epoch 83/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.1801 - accuracy: 0.7686
    Epoch 84/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.3144 - accuracy: 0.7722
    Epoch 85/6000
    35/35 [==============================] - 0s 471us/step - loss: 18.6149 - accuracy: 0.7196
    Epoch 86/6000
    35/35 [==============================] - 0s 441us/step - loss: 17.6336 - accuracy: 0.7405
    Epoch 87/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.5094 - accuracy: 0.7486
    Epoch 88/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2231 - accuracy: 0.7722
    Epoch 89/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7586 - accuracy: 0.7822
    Epoch 90/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0169 - accuracy: 0.8031
    Epoch 91/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.8612 - accuracy: 0.7613
    Epoch 92/6000
    35/35 [==============================] - 0s 412us/step - loss: 3.4757 - accuracy: 0.7713
    Epoch 93/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.4097 - accuracy: 0.7849
    Epoch 94/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.5222 - accuracy: 0.7958
    Epoch 95/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.5011 - accuracy: 0.7332
    Epoch 96/6000
    35/35 [==============================] - 0s 471us/step - loss: 22.8488 - accuracy: 0.7223
    Epoch 97/6000
    35/35 [==============================] - 0s 500us/step - loss: 6.8344 - accuracy: 0.7704
    Epoch 98/6000
    35/35 [==============================] - 0s 529us/step - loss: 18.0335 - accuracy: 0.7541
    Epoch 99/6000
    35/35 [==============================] - 0s 500us/step - loss: 7.4647 - accuracy: 0.7523
    Epoch 100/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.0455 - accuracy: 0.7931
    Epoch 101/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4369 - accuracy: 0.7650
    Epoch 102/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.8936 - accuracy: 0.7641
    Epoch 103/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.5442 - accuracy: 0.7459
    Epoch 104/6000
    35/35 [==============================] - 0s 470us/step - loss: 2.4748 - accuracy: 0.7895
    Epoch 105/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.8317 - accuracy: 0.7632
    Epoch 106/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.8707 - accuracy: 0.7659
    Epoch 107/6000
    35/35 [==============================] - 0s 529us/step - loss: 14.6616 - accuracy: 0.7623
    Epoch 108/6000
    35/35 [==============================] - 0s 441us/step - loss: 19.2367 - accuracy: 0.7323
    Epoch 109/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.6694 - accuracy: 0.7577
    Epoch 110/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7336 - accuracy: 0.7804
    Epoch 111/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.0982 - accuracy: 0.7541
    Epoch 112/6000
    35/35 [==============================] - 0s 441us/step - loss: 11.1078 - accuracy: 0.7904
    Epoch 113/6000
    35/35 [==============================] - 0s 470us/step - loss: 7.1662 - accuracy: 0.7532
    Epoch 114/6000
    35/35 [==============================] - 0s 471us/step - loss: 8.6123 - accuracy: 0.7387
    Epoch 115/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.7064 - accuracy: 0.7495
    Epoch 116/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.5137 - accuracy: 0.7641
    Epoch 117/6000
    35/35 [==============================] - 0s 412us/step - loss: 7.4366 - accuracy: 0.7477
    Epoch 118/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.5197 - accuracy: 0.7604
    Epoch 119/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8530 - accuracy: 0.7623
    Epoch 120/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.3160 - accuracy: 0.7632
    Epoch 121/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.6095 - accuracy: 0.7650
    Epoch 122/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4262 - accuracy: 0.7659
    Epoch 123/6000
    35/35 [==============================] - 0s 441us/step - loss: 12.8789 - accuracy: 0.7305
    Epoch 124/6000
    35/35 [==============================] - 0s 471us/step - loss: 10.0823 - accuracy: 0.7532
    Epoch 125/6000
    35/35 [==============================] - 0s 500us/step - loss: 13.9329 - accuracy: 0.7486
    Epoch 126/6000
    35/35 [==============================] - 0s 471us/step - loss: 29.9708 - accuracy: 0.7341
    Epoch 127/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.3589 - accuracy: 0.7650
    Epoch 128/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.0132 - accuracy: 0.7550
    Epoch 129/6000
    35/35 [==============================] - 0s 441us/step - loss: 11.6910 - accuracy: 0.7459
    Epoch 130/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9919 - accuracy: 0.8258
    Epoch 131/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.0615 - accuracy: 0.7550
    Epoch 132/6000
    35/35 [==============================] - 0s 500us/step - loss: 5.8490 - accuracy: 0.7550
    Epoch 133/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0228 - accuracy: 0.8058
    Epoch 134/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.6738 - accuracy: 0.7804
    Epoch 135/6000
    35/35 [==============================] - 0s 500us/step - loss: 6.1687 - accuracy: 0.7541
    Epoch 136/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3022 - accuracy: 0.7840
    Epoch 137/6000
    35/35 [==============================] - 0s 471us/step - loss: 17.7663 - accuracy: 0.7568
    Epoch 138/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.3601 - accuracy: 0.7722
    Epoch 139/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.6150 - accuracy: 0.7795
    Epoch 140/6000
    35/35 [==============================] - 0s 471us/step - loss: 18.5114 - accuracy: 0.7532
    Epoch 141/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.5009 - accuracy: 0.7704
    Epoch 142/6000
    35/35 [==============================] - 0s 529us/step - loss: 4.4871 - accuracy: 0.7668
    Epoch 143/6000
    35/35 [==============================] - 0s 559us/step - loss: 3.1039 - accuracy: 0.7904
    Epoch 144/6000
    35/35 [==============================] - 0s 471us/step - loss: 16.9926 - accuracy: 0.7223
    Epoch 145/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.1878 - accuracy: 0.7477
    Epoch 146/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8958 - accuracy: 0.7704
    Epoch 147/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.2293 - accuracy: 0.7613
    Epoch 148/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.4852 - accuracy: 0.7414
    Epoch 149/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.0270 - accuracy: 0.7432
    Epoch 150/6000
    35/35 [==============================] - 0s 470us/step - loss: 17.5186 - accuracy: 0.7468
    Epoch 151/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.7510 - accuracy: 0.7668
    Epoch 152/6000
    35/35 [==============================] - 0s 529us/step - loss: 4.9649 - accuracy: 0.7659
    Epoch 153/6000
    35/35 [==============================] - 0s 471us/step - loss: 12.5490 - accuracy: 0.7550
    Epoch 154/6000
    35/35 [==============================] - 0s 500us/step - loss: 9.0741 - accuracy: 0.7559
    Epoch 155/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.0477 - accuracy: 0.7668
    Epoch 156/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.2289 - accuracy: 0.7759
    Epoch 157/6000
    35/35 [==============================] - 0s 500us/step - loss: 7.6051 - accuracy: 0.7296
    Epoch 158/6000
    35/35 [==============================] - 0s 559us/step - loss: 9.2619 - accuracy: 0.7559
    Epoch 159/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.6960 - accuracy: 0.7541
    Epoch 160/6000
    35/35 [==============================] - 0s 500us/step - loss: 9.5976 - accuracy: 0.7768
    Epoch 161/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.1879 - accuracy: 0.7849
    Epoch 162/6000
    35/35 [==============================] - 0s 500us/step - loss: 5.1049 - accuracy: 0.7505
    Epoch 163/6000
    35/35 [==============================] - 0s 471us/step - loss: 15.1671 - accuracy: 0.7350
    Epoch 164/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4916 - accuracy: 0.7904
    Epoch 165/6000
    35/35 [==============================] - 0s 470us/step - loss: 2.5349 - accuracy: 0.7958
    Epoch 166/6000
    35/35 [==============================] - 0s 441us/step - loss: 11.0692 - accuracy: 0.7577
    Epoch 167/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.8327 - accuracy: 0.7913
    Epoch 168/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.7661 - accuracy: 0.7514
    Epoch 169/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.4194 - accuracy: 0.7287
    Epoch 170/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.1693 - accuracy: 0.7514
    Epoch 171/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0331 - accuracy: 0.8085
    Epoch 172/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.2677 - accuracy: 0.7913
    Epoch 173/6000
    35/35 [==============================] - 0s 471us/step - loss: 10.0114 - accuracy: 0.7514
    Epoch 174/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.5434 - accuracy: 0.7532
    Epoch 175/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.3820 - accuracy: 0.7541
    Epoch 176/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.6923 - accuracy: 0.7641
    Epoch 177/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.4783 - accuracy: 0.7650
    Epoch 178/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.5129 - accuracy: 0.7613
    Epoch 179/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3078 - accuracy: 0.7804
    Epoch 180/6000
    35/35 [==============================] - 0s 470us/step - loss: 3.6724 - accuracy: 0.7967
    Epoch 181/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0628 - accuracy: 0.7976
    Epoch 182/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3330 - accuracy: 0.7759
    Epoch 183/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6935 - accuracy: 0.8094
    Epoch 184/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.9634 - accuracy: 0.7704
    Epoch 185/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.9974 - accuracy: 0.7786
    Epoch 186/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.4567 - accuracy: 0.7795
    Epoch 187/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.5455 - accuracy: 0.7477
    Epoch 188/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4071 - accuracy: 0.8022
    Epoch 189/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.4204 - accuracy: 0.7532
    Epoch 190/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.9256 - accuracy: 0.7505
    Epoch 191/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0675 - accuracy: 0.7713
    Epoch 192/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.1104 - accuracy: 0.7795
    Epoch 193/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3905 - accuracy: 0.7704
    Epoch 194/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5684 - accuracy: 0.7831
    Epoch 195/6000
    35/35 [==============================] - 0s 441us/step - loss: 17.0824 - accuracy: 0.7305
    Epoch 196/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8858 - accuracy: 0.7641
    Epoch 197/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.6623 - accuracy: 0.7523
    Epoch 198/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5572 - accuracy: 0.7713
    Epoch 199/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4852 - accuracy: 0.8094
    Epoch 200/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.3341 - accuracy: 0.7913
    Epoch 201/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4901 - accuracy: 0.7659
    Epoch 202/6000
    35/35 [==============================] - 0s 412us/step - loss: 4.1418 - accuracy: 0.7940
    Epoch 203/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.9294 - accuracy: 0.7495
    Epoch 204/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.8783 - accuracy: 0.7586
    Epoch 205/6000
    35/35 [==============================] - 0s 471us/step - loss: 14.1604 - accuracy: 0.7541
    Epoch 206/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.8877 - accuracy: 0.7505
    Epoch 207/6000
    35/35 [==============================] - 0s 471us/step - loss: 13.0449 - accuracy: 0.7514
    Epoch 208/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.4430 - accuracy: 0.7704
    Epoch 209/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9822 - accuracy: 0.7740
    Epoch 210/6000
    35/35 [==============================] - 0s 500us/step - loss: 5.9576 - accuracy: 0.7677
    Epoch 211/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9729 - accuracy: 0.8040
    Epoch 212/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.9604 - accuracy: 0.7613
    Epoch 213/6000
    35/35 [==============================] - 0s 441us/step - loss: 14.1036 - accuracy: 0.7641
    Epoch 214/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5890 - accuracy: 0.7722
    Epoch 215/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9353 - accuracy: 0.7913
    Epoch 216/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.1086 - accuracy: 0.7877
    Epoch 217/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5236 - accuracy: 0.8076
    Epoch 218/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.3887 - accuracy: 0.7677
    Epoch 219/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.8677 - accuracy: 0.7731
    Epoch 220/6000
    35/35 [==============================] - 0s 441us/step - loss: 15.4671 - accuracy: 0.7604
    Epoch 221/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7518 - accuracy: 0.7849
    Epoch 222/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.4423 - accuracy: 0.7595
    Epoch 223/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.3500 - accuracy: 0.7613
    Epoch 224/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.9216 - accuracy: 0.7486
    Epoch 225/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6333 - accuracy: 0.7931
    Epoch 226/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.7202 - accuracy: 0.7586
    Epoch 227/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.0115 - accuracy: 0.7668
    Epoch 228/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.6326 - accuracy: 0.8113
    Epoch 229/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4516 - accuracy: 0.7686
    Epoch 230/6000
    35/35 [==============================] - 0s 412us/step - loss: 10.4287 - accuracy: 0.7495
    Epoch 231/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9069 - accuracy: 0.7686
    Epoch 232/6000
    35/35 [==============================] - 0s 412us/step - loss: 19.9460 - accuracy: 0.7250
    Epoch 233/6000
    35/35 [==============================] - 0s 441us/step - loss: 12.6285 - accuracy: 0.7650
    Epoch 234/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3971 - accuracy: 0.7849
    Epoch 235/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.3113 - accuracy: 0.7532
    Epoch 236/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.2284 - accuracy: 0.7722
    Epoch 237/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.6768 - accuracy: 0.7568
    Epoch 238/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5436 - accuracy: 0.7922
    Epoch 239/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.1449 - accuracy: 0.7677
    Epoch 240/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.2102 - accuracy: 0.7759
    Epoch 241/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2653 - accuracy: 0.8040
    Epoch 242/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.5975 - accuracy: 0.7623
    Epoch 243/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.9516 - accuracy: 0.7831
    Epoch 244/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4540 - accuracy: 0.8094
    Epoch 245/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.5138 - accuracy: 0.7768
    Epoch 246/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.3512 - accuracy: 0.7550
    Epoch 247/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9442 - accuracy: 0.8058
    Epoch 248/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.4509 - accuracy: 0.7641
    Epoch 249/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2122 - accuracy: 0.8122
    Epoch 250/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.5757 - accuracy: 0.7477
    Epoch 251/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9015 - accuracy: 0.7868
    Epoch 252/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.4022 - accuracy: 0.7759
    Epoch 253/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.5338 - accuracy: 0.7468
    Epoch 254/6000
    35/35 [==============================] - 0s 471us/step - loss: 17.5127 - accuracy: 0.7414
    Epoch 255/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.1738 - accuracy: 0.7795
    Epoch 256/6000
    35/35 [==============================] - 0s 500us/step - loss: 8.4286 - accuracy: 0.7740
    Epoch 257/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.7329 - accuracy: 0.7886
    Epoch 258/6000
    35/35 [==============================] - 0s 412us/step - loss: 6.8310 - accuracy: 0.7359
    Epoch 259/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9733 - accuracy: 0.7713
    Epoch 260/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.2388 - accuracy: 0.7822
    Epoch 261/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.6063 - accuracy: 0.7677
    Epoch 262/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.0454 - accuracy: 0.7795
    Epoch 263/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.3083 - accuracy: 0.7677
    Epoch 264/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0918 - accuracy: 0.8094
    Epoch 265/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9323 - accuracy: 0.8113
    Epoch 266/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.6247 - accuracy: 0.7577
    Epoch 267/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9834 - accuracy: 0.7768
    Epoch 268/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.1162 - accuracy: 0.7840
    Epoch 269/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8641 - accuracy: 0.7777
    Epoch 270/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.4842 - accuracy: 0.7940
    Epoch 271/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9252 - accuracy: 0.7985
    Epoch 272/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.2480 - accuracy: 0.7849
    Epoch 273/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8764 - accuracy: 0.8022
    Epoch 274/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.4421 - accuracy: 0.7604
    Epoch 275/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.2423 - accuracy: 0.7813
    Epoch 276/6000
    35/35 [==============================] - 0s 412us/step - loss: 3.2214 - accuracy: 0.8004
    Epoch 277/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8072 - accuracy: 0.7759
    Epoch 278/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.4735 - accuracy: 0.7650
    Epoch 279/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.0912 - accuracy: 0.7913
    Epoch 280/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3372 - accuracy: 0.7931
    Epoch 281/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.6304 - accuracy: 0.7858
    Epoch 282/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.1200 - accuracy: 0.7641
    Epoch 283/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0460 - accuracy: 0.8049
    Epoch 284/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.2132 - accuracy: 0.7722
    Epoch 285/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.1815 - accuracy: 0.7922
    Epoch 286/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7368 - accuracy: 0.7759
    Epoch 287/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.4145 - accuracy: 0.7858
    Epoch 288/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2228 - accuracy: 0.7840
    Epoch 289/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.8511 - accuracy: 0.7604
    Epoch 290/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5111 - accuracy: 0.7713
    Epoch 291/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5474 - accuracy: 0.7985
    Epoch 292/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.9657 - accuracy: 0.7423
    Epoch 293/6000
    35/35 [==============================] - 0s 500us/step - loss: 6.5461 - accuracy: 0.7677
    Epoch 294/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.4008 - accuracy: 0.7604
    Epoch 295/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.3068 - accuracy: 0.7623
    Epoch 296/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9586 - accuracy: 0.7922
    Epoch 297/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3926 - accuracy: 0.7858
    Epoch 298/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.3303 - accuracy: 0.7985
    Epoch 299/6000
    35/35 [==============================] - 0s 588us/step - loss: 2.9394 - accuracy: 0.7949
    Epoch 300/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.6813 - accuracy: 0.7976
    Epoch 301/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.9792 - accuracy: 0.7958
    Epoch 302/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.7406 - accuracy: 0.7777
    Epoch 303/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.6397 - accuracy: 0.7931
    Epoch 304/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1778 - accuracy: 0.7877
    Epoch 305/6000
    35/35 [==============================] - 0s 471us/step - loss: 16.0084 - accuracy: 0.7414
    Epoch 306/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.1323 - accuracy: 0.7849
    Epoch 307/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.9092 - accuracy: 0.7759
    Epoch 308/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.2406 - accuracy: 0.7795
    Epoch 309/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.3587 - accuracy: 0.7777
    Epoch 310/6000
    35/35 [==============================] - 0s 471us/step - loss: 8.4080 - accuracy: 0.7414
    Epoch 311/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7778 - accuracy: 0.7623
    Epoch 312/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.2816 - accuracy: 0.7786
    Epoch 313/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.1602 - accuracy: 0.7586
    Epoch 314/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.2548 - accuracy: 0.7731
    Epoch 315/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4303 - accuracy: 0.7840
    Epoch 316/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7240 - accuracy: 0.8076
    Epoch 317/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.0377 - accuracy: 0.7659
    Epoch 318/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9514 - accuracy: 0.7768
    Epoch 319/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.1811 - accuracy: 0.7550
    Epoch 320/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.4389 - accuracy: 0.7468
    Epoch 321/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8978 - accuracy: 0.8113
    Epoch 322/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.4918 - accuracy: 0.7668
    Epoch 323/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9709 - accuracy: 0.7904
    Epoch 324/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.3079 - accuracy: 0.7740
    Epoch 325/6000
    35/35 [==============================] - 0s 471us/step - loss: 14.5755 - accuracy: 0.7613
    Epoch 326/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.1410 - accuracy: 0.7659
    Epoch 327/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.4406 - accuracy: 0.7940
    Epoch 328/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.8765 - accuracy: 0.7541
    Epoch 329/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.6231 - accuracy: 0.7477
    Epoch 330/6000
    35/35 [==============================] - 0s 441us/step - loss: 14.5276 - accuracy: 0.7577
    Epoch 331/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7384 - accuracy: 0.8203
    Epoch 332/6000
    35/35 [==============================] - 0s 412us/step - loss: 3.4436 - accuracy: 0.7868
    Epoch 333/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.8620 - accuracy: 0.7650
    Epoch 334/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6745 - accuracy: 0.7858
    Epoch 335/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.4082 - accuracy: 0.7695
    Epoch 336/6000
    35/35 [==============================] - 0s 529us/step - loss: 5.3422 - accuracy: 0.7677
    Epoch 337/6000
    35/35 [==============================] - 0s 529us/step - loss: 3.4751 - accuracy: 0.7831
    Epoch 338/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.1051 - accuracy: 0.7750
    Epoch 339/6000
    35/35 [==============================] - 0s 618us/step - loss: 12.2431 - accuracy: 0.7541
    Epoch 340/6000
    35/35 [==============================] - 0s 500us/step - loss: 6.7733 - accuracy: 0.7677
    Epoch 341/6000
    35/35 [==============================] - 0s 500us/step - loss: 13.1071 - accuracy: 0.7577
    Epoch 342/6000
    35/35 [==============================] - 0s 471us/step - loss: 10.8498 - accuracy: 0.7559
    Epoch 343/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.2586 - accuracy: 0.7840
    Epoch 344/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.6239 - accuracy: 0.7886
    Epoch 345/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.8242 - accuracy: 0.7868
    Epoch 346/6000
    35/35 [==============================] - 0s 500us/step - loss: 8.8157 - accuracy: 0.7595
    Epoch 347/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.3482 - accuracy: 0.7613
    Epoch 348/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8146 - accuracy: 0.7904
    Epoch 349/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.3508 - accuracy: 0.7804
    Epoch 350/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.9808 - accuracy: 0.7686
    Epoch 351/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.8492 - accuracy: 0.7813
    Epoch 352/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7614 - accuracy: 0.8122
    Epoch 353/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5600 - accuracy: 0.7759
    Epoch 354/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7542 - accuracy: 0.7677
    Epoch 355/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3972 - accuracy: 0.7904
    Epoch 356/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.6534 - accuracy: 0.7650
    Epoch 357/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2613 - accuracy: 0.8194
    Epoch 358/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1604 - accuracy: 0.8358
    Epoch 359/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.6037 - accuracy: 0.7632
    Epoch 360/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.3423 - accuracy: 0.7731
    Epoch 361/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4070 - accuracy: 0.7722
    Epoch 362/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.5313 - accuracy: 0.7541
    Epoch 363/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.0528 - accuracy: 0.7759
    Epoch 364/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8568 - accuracy: 0.7840
    Epoch 365/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3035 - accuracy: 0.7995
    Epoch 366/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2507 - accuracy: 0.7886
    Epoch 367/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.9595 - accuracy: 0.7886
    Epoch 368/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.0075 - accuracy: 0.7849
    Epoch 369/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5413 - accuracy: 0.8113
    Epoch 370/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4001 - accuracy: 0.8004
    Epoch 371/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.8039 - accuracy: 0.7541
    Epoch 372/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8216 - accuracy: 0.7768
    Epoch 373/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7125 - accuracy: 0.8285
    Epoch 374/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1297 - accuracy: 0.7949
    Epoch 375/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.7705 - accuracy: 0.7686
    Epoch 376/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9280 - accuracy: 0.8122
    Epoch 377/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6037 - accuracy: 0.8321
    Epoch 378/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.6854 - accuracy: 0.7786
    Epoch 379/6000
    35/35 [==============================] - 0s 412us/step - loss: 3.0704 - accuracy: 0.7868
    Epoch 380/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.8100 - accuracy: 0.7486
    Epoch 381/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.4456 - accuracy: 0.8013
    Epoch 382/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9957 - accuracy: 0.8131
    Epoch 383/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6580 - accuracy: 0.7877
    Epoch 384/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.1728 - accuracy: 0.7804
    Epoch 385/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5849 - accuracy: 0.7985
    Epoch 386/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9982 - accuracy: 0.7822
    Epoch 387/6000
    35/35 [==============================] - 0s 412us/step - loss: 4.4045 - accuracy: 0.7849
    Epoch 388/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5725 - accuracy: 0.8158
    Epoch 389/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8948 - accuracy: 0.8022
    Epoch 390/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.2131 - accuracy: 0.8031
    Epoch 391/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2989 - accuracy: 0.7931
    Epoch 392/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3010 - accuracy: 0.8122
    Epoch 393/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2520 - accuracy: 0.8031
    Epoch 394/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2960 - accuracy: 0.7759
    Epoch 395/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.4851 - accuracy: 0.7686
    Epoch 396/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3187 - accuracy: 0.7786
    Epoch 397/6000
    35/35 [==============================] - 0s 412us/step - loss: 8.9204 - accuracy: 0.7731
    Epoch 398/6000
    35/35 [==============================] - 0s 500us/step - loss: 9.6243 - accuracy: 0.7668
    Epoch 399/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.9938 - accuracy: 0.7432
    Epoch 400/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0780 - accuracy: 0.8230
    Epoch 401/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3923 - accuracy: 0.7958
    Epoch 402/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9842 - accuracy: 0.7886
    Epoch 403/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9357 - accuracy: 0.7877
    Epoch 404/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8608 - accuracy: 0.8031
    Epoch 405/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.4872 - accuracy: 0.7731
    Epoch 406/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.3306 - accuracy: 0.7886
    Epoch 407/6000
    35/35 [==============================] - 0s 441us/step - loss: 17.5379 - accuracy: 0.7305
    Epoch 408/6000
    35/35 [==============================] - 0s 441us/step - loss: 17.8262 - accuracy: 0.7568
    Epoch 409/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.4551 - accuracy: 0.7595
    Epoch 410/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7354 - accuracy: 0.8085
    Epoch 411/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.6056 - accuracy: 0.7477
    Epoch 412/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2012 - accuracy: 0.7985
    Epoch 413/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5965 - accuracy: 0.7840
    Epoch 414/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7170 - accuracy: 0.8258
    Epoch 415/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.0575 - accuracy: 0.7886
    Epoch 416/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0571 - accuracy: 0.8049
    Epoch 417/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.4434 - accuracy: 0.7514
    Epoch 418/6000
    35/35 [==============================] - 0s 412us/step - loss: 5.8295 - accuracy: 0.7650
    Epoch 419/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.8027 - accuracy: 0.7377
    Epoch 420/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.5225 - accuracy: 0.7804
    Epoch 421/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7935 - accuracy: 0.7877
    Epoch 422/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8030 - accuracy: 0.8103
    Epoch 423/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9351 - accuracy: 0.7949
    Epoch 424/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3497 - accuracy: 0.8094
    Epoch 425/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.3089 - accuracy: 0.8267
    Epoch 426/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4525 - accuracy: 0.8058
    Epoch 427/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.0537 - accuracy: 0.8113
    Epoch 428/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.5919 - accuracy: 0.7713
    Epoch 429/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1287 - accuracy: 0.8094
    Epoch 430/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.9856 - accuracy: 0.7740
    Epoch 431/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.3395 - accuracy: 0.7677
    Epoch 432/6000
    35/35 [==============================] - 0s 529us/step - loss: 5.5015 - accuracy: 0.7477
    Epoch 433/6000
    35/35 [==============================] - 0s 559us/step - loss: 5.8363 - accuracy: 0.7786
    Epoch 434/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2427 - accuracy: 0.8122
    Epoch 435/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8551 - accuracy: 0.8185
    Epoch 436/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.2541 - accuracy: 0.8022
    Epoch 437/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0303 - accuracy: 0.7985
    Epoch 438/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.9678 - accuracy: 0.7532
    Epoch 439/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.9347 - accuracy: 0.7868
    Epoch 440/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.6536 - accuracy: 0.7568
    Epoch 441/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.9550 - accuracy: 0.7586
    Epoch 442/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.9020 - accuracy: 0.7731
    Epoch 443/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6570 - accuracy: 0.8167
    Epoch 444/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5925 - accuracy: 0.7958
    Epoch 445/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5274 - accuracy: 0.7940
    Epoch 446/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0675 - accuracy: 0.7650
    Epoch 447/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.6488 - accuracy: 0.7868
    Epoch 448/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7272 - accuracy: 0.8103
    Epoch 449/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2631 - accuracy: 0.8358
    Epoch 450/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2465 - accuracy: 0.8385
    Epoch 451/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4310 - accuracy: 0.8013
    Epoch 452/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4350 - accuracy: 0.7623
    Epoch 453/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7658 - accuracy: 0.8040
    Epoch 454/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.4975 - accuracy: 0.7786
    Epoch 455/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.7912 - accuracy: 0.7495
    Epoch 456/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4580 - accuracy: 0.7659
    Epoch 457/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.9833 - accuracy: 0.7804
    Epoch 458/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2394 - accuracy: 0.7940
    Epoch 459/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0222 - accuracy: 0.8004
    Epoch 460/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.2923 - accuracy: 0.7523
    Epoch 461/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.6740 - accuracy: 0.7595
    Epoch 462/6000
    35/35 [==============================] - 0s 471us/step - loss: 8.0095 - accuracy: 0.7495
    Epoch 463/6000
    35/35 [==============================] - 0s 412us/step - loss: 5.0487 - accuracy: 0.8113
    Epoch 464/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9193 - accuracy: 0.8194
    Epoch 465/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9581 - accuracy: 0.7886
    Epoch 466/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.8485 - accuracy: 0.7804
    Epoch 467/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1028 - accuracy: 0.8185
    Epoch 468/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2225 - accuracy: 0.8004
    Epoch 469/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.8222 - accuracy: 0.8004
    Epoch 470/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6295 - accuracy: 0.8249
    Epoch 471/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5944 - accuracy: 0.8212
    Epoch 472/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.9160 - accuracy: 0.7722
    Epoch 473/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9535 - accuracy: 0.8022
    Epoch 474/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5537 - accuracy: 0.8203
    Epoch 475/6000
    35/35 [==============================] - 0s 412us/step - loss: 6.4753 - accuracy: 0.7332
    Epoch 476/6000
    35/35 [==============================] - 0s 441us/step - loss: 12.0949 - accuracy: 0.7704
    Epoch 477/6000
    35/35 [==============================] - 0s 412us/step - loss: 3.4691 - accuracy: 0.7868
    Epoch 478/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7201 - accuracy: 0.7759
    Epoch 479/6000
    35/35 [==============================] - 0s 412us/step - loss: 5.3145 - accuracy: 0.7632
    Epoch 480/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.2314 - accuracy: 0.7913
    Epoch 481/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.7233 - accuracy: 0.7777
    Epoch 482/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8098 - accuracy: 0.8058
    Epoch 483/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5375 - accuracy: 0.8004
    Epoch 484/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9528 - accuracy: 0.7940
    Epoch 485/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.3050 - accuracy: 0.7967
    Epoch 486/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.3674 - accuracy: 0.7641
    Epoch 487/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7486 - accuracy: 0.8058
    Epoch 488/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2327 - accuracy: 0.7877
    Epoch 489/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2612 - accuracy: 0.8167
    Epoch 490/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.4494 - accuracy: 0.7623
    Epoch 491/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7720 - accuracy: 0.7795
    Epoch 492/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.5918 - accuracy: 0.7985
    Epoch 493/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7570 - accuracy: 0.8049
    Epoch 494/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1433 - accuracy: 0.8031
    Epoch 495/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9621 - accuracy: 0.8031
    Epoch 496/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.3484 - accuracy: 0.8448
    Epoch 497/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4165 - accuracy: 0.7868
    Epoch 498/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2348 - accuracy: 0.8267
    Epoch 499/6000
    35/35 [==============================] - 0s 441us/step - loss: 10.1006 - accuracy: 0.7405
    Epoch 500/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.1638 - accuracy: 0.8085
    Epoch 501/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.1467 - accuracy: 0.7722
    Epoch 502/6000
    35/35 [==============================] - 0s 471us/step - loss: 9.1284 - accuracy: 0.7595
    Epoch 503/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.5354 - accuracy: 0.7613
    Epoch 504/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.4283 - accuracy: 0.7368
    Epoch 505/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.9661 - accuracy: 0.7459
    Epoch 506/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4859 - accuracy: 0.7813
    Epoch 507/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1934 - accuracy: 0.7913
    Epoch 508/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6565 - accuracy: 0.8212
    Epoch 509/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7374 - accuracy: 0.8004
    Epoch 510/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6621 - accuracy: 0.7849
    Epoch 511/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.0157 - accuracy: 0.7759
    Epoch 512/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8294 - accuracy: 0.8067
    Epoch 513/6000
    35/35 [==============================] - 0s 412us/step - loss: 3.2027 - accuracy: 0.8122
    Epoch 514/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2747 - accuracy: 0.8058
    Epoch 515/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2687 - accuracy: 0.8049
    Epoch 516/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2324 - accuracy: 0.7877
    Epoch 517/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5387 - accuracy: 0.8194
    Epoch 518/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3159 - accuracy: 0.7786
    Epoch 519/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.8317 - accuracy: 0.7940
    Epoch 520/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.4741 - accuracy: 0.7868
    Epoch 521/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1920 - accuracy: 0.7985
    Epoch 522/6000
    35/35 [==============================] - 0s 412us/step - loss: 5.0991 - accuracy: 0.7722
    Epoch 523/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2968 - accuracy: 0.8004
    Epoch 524/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.4657 - accuracy: 0.7595
    Epoch 525/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.4073 - accuracy: 0.7958
    Epoch 526/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.7841 - accuracy: 0.7786
    Epoch 527/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.8236 - accuracy: 0.7459
    Epoch 528/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7194 - accuracy: 0.8212
    Epoch 529/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4748 - accuracy: 0.7695
    Epoch 530/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4361 - accuracy: 0.8085
    Epoch 531/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1960 - accuracy: 0.7895
    Epoch 532/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.3940 - accuracy: 0.7722
    Epoch 533/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4682 - accuracy: 0.7985
    Epoch 534/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5667 - accuracy: 0.8339
    Epoch 535/6000
    35/35 [==============================] - 0s 441us/step - loss: 13.4275 - accuracy: 0.7260
    Epoch 536/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4572 - accuracy: 0.7976
    Epoch 537/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.4536 - accuracy: 0.7795
    Epoch 538/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.0186 - accuracy: 0.7532
    Epoch 539/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2422 - accuracy: 0.8022
    Epoch 540/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.9045 - accuracy: 0.7849
    Epoch 541/6000
    35/35 [==============================] - 0s 471us/step - loss: 12.9806 - accuracy: 0.7441
    Epoch 542/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.7014 - accuracy: 0.7731
    Epoch 543/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3080 - accuracy: 0.8094
    Epoch 544/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9091 - accuracy: 0.7795
    Epoch 545/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7266 - accuracy: 0.8176
    Epoch 546/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1552 - accuracy: 0.8240
    Epoch 547/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8535 - accuracy: 0.8149
    Epoch 548/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9164 - accuracy: 0.8113
    Epoch 549/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6909 - accuracy: 0.8158
    Epoch 550/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5730 - accuracy: 0.7976
    Epoch 551/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3615 - accuracy: 0.8049
    Epoch 552/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2674 - accuracy: 0.8067
    Epoch 553/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.6902 - accuracy: 0.7822
    Epoch 554/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.8318 - accuracy: 0.7750
    Epoch 555/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6837 - accuracy: 0.7913
    Epoch 556/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.7452 - accuracy: 0.7777
    Epoch 557/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7257 - accuracy: 0.7831
    Epoch 558/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7646 - accuracy: 0.7858
    Epoch 559/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8712 - accuracy: 0.7550
    Epoch 560/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9447 - accuracy: 0.7877
    Epoch 561/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.3987 - accuracy: 0.7786
    Epoch 562/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2137 - accuracy: 0.8067
    Epoch 563/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8502 - accuracy: 0.7940
    Epoch 564/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.2223 - accuracy: 0.7804
    Epoch 565/6000
    35/35 [==============================] - 0s 529us/step - loss: 3.0630 - accuracy: 0.7949
    Epoch 566/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5904 - accuracy: 0.8040
    Epoch 567/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9450 - accuracy: 0.8094
    Epoch 568/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.5369 - accuracy: 0.7650
    Epoch 569/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0561 - accuracy: 0.8049
    Epoch 570/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8559 - accuracy: 0.8194
    Epoch 571/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8493 - accuracy: 0.8167
    Epoch 572/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7518 - accuracy: 0.8040
    Epoch 573/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7256 - accuracy: 0.7922
    Epoch 574/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1290 - accuracy: 0.7995
    Epoch 575/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7274 - accuracy: 0.7868
    Epoch 576/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.6890 - accuracy: 0.7604
    Epoch 577/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.8495 - accuracy: 0.7731
    Epoch 578/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9227 - accuracy: 0.8240
    Epoch 579/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3976 - accuracy: 0.8076
    Epoch 580/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4526 - accuracy: 0.7586
    Epoch 581/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.6723 - accuracy: 0.7713
    Epoch 582/6000
    35/35 [==============================] - 0s 441us/step - loss: 11.8221 - accuracy: 0.7468
    Epoch 583/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3516 - accuracy: 0.7822
    Epoch 584/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.1705 - accuracy: 0.7822
    Epoch 585/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.8909 - accuracy: 0.7831
    Epoch 586/6000
    35/35 [==============================] - 0s 441us/step - loss: 14.4265 - accuracy: 0.7559
    Epoch 587/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.0720 - accuracy: 0.7740
    Epoch 588/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5020 - accuracy: 0.8031
    Epoch 589/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8889 - accuracy: 0.7740
    Epoch 590/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1981 - accuracy: 0.7904
    Epoch 591/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0927 - accuracy: 0.8031
    Epoch 592/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7635 - accuracy: 0.7795
    Epoch 593/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1030 - accuracy: 0.8049
    Epoch 594/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3412 - accuracy: 0.7958
    Epoch 595/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5979 - accuracy: 0.8022
    Epoch 596/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.0665 - accuracy: 0.7958
    Epoch 597/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.7379 - accuracy: 0.7523
    Epoch 598/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.8995 - accuracy: 0.7695
    Epoch 599/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4339 - accuracy: 0.8058
    Epoch 600/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5922 - accuracy: 0.8267
    Epoch 601/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4998 - accuracy: 0.7931
    Epoch 602/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3386 - accuracy: 0.7704
    Epoch 603/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1950 - accuracy: 0.8321
    Epoch 604/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9736 - accuracy: 0.7750
    Epoch 605/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7307 - accuracy: 0.7813
    Epoch 606/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1933 - accuracy: 0.7922
    Epoch 607/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5825 - accuracy: 0.7913
    Epoch 608/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7861 - accuracy: 0.7868
    Epoch 609/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.7386 - accuracy: 0.8185
    Epoch 610/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6118 - accuracy: 0.8285
    Epoch 611/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9512 - accuracy: 0.8022
    Epoch 612/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.3770 - accuracy: 0.7495
    Epoch 613/6000
    35/35 [==============================] - 0s 441us/step - loss: 13.7533 - accuracy: 0.7332
    Epoch 614/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5085 - accuracy: 0.8113
    Epoch 615/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8323 - accuracy: 0.8067
    Epoch 616/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6827 - accuracy: 0.7995
    Epoch 617/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2704 - accuracy: 0.8339
    Epoch 618/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.9637 - accuracy: 0.7849
    Epoch 619/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0659 - accuracy: 0.7623
    Epoch 620/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7325 - accuracy: 0.8212
    Epoch 621/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1111 - accuracy: 0.7886
    Epoch 622/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.5108 - accuracy: 0.7759
    Epoch 623/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.4304 - accuracy: 0.8013
    Epoch 624/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3941 - accuracy: 0.8076
    Epoch 625/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.5363 - accuracy: 0.7713
    Epoch 626/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.5957 - accuracy: 0.7641
    Epoch 627/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.7171 - accuracy: 0.7632
    Epoch 628/6000
    35/35 [==============================] - 0s 412us/step - loss: 5.0089 - accuracy: 0.7713
    Epoch 629/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6493 - accuracy: 0.8131
    Epoch 630/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1610 - accuracy: 0.7940
    Epoch 631/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2502 - accuracy: 0.8385
    Epoch 632/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.1610 - accuracy: 0.7976
    Epoch 633/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8484 - accuracy: 0.7967
    Epoch 634/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4423 - accuracy: 0.8149
    Epoch 635/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.4314 - accuracy: 0.7713
    Epoch 636/6000
    35/35 [==============================] - 0s 441us/step - loss: 15.0333 - accuracy: 0.7323
    Epoch 637/6000
    35/35 [==============================] - 0s 441us/step - loss: 11.9561 - accuracy: 0.7559
    Epoch 638/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.3241 - accuracy: 0.7368
    Epoch 639/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3061 - accuracy: 0.8430
    Epoch 640/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6402 - accuracy: 0.7913
    Epoch 641/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3649 - accuracy: 0.8031
    Epoch 642/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0429 - accuracy: 0.8076
    Epoch 643/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.4779 - accuracy: 0.7632
    Epoch 644/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6126 - accuracy: 0.7931
    Epoch 645/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3845 - accuracy: 0.7985
    Epoch 646/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7458 - accuracy: 0.8131
    Epoch 647/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3347 - accuracy: 0.8276
    Epoch 648/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.3856 - accuracy: 0.7967
    Epoch 649/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5560 - accuracy: 0.8230
    Epoch 650/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.7721 - accuracy: 0.7550
    Epoch 651/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9654 - accuracy: 0.8067
    Epoch 652/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2717 - accuracy: 0.8258
    Epoch 653/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4100 - accuracy: 0.8367
    Epoch 654/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9279 - accuracy: 0.8131
    Epoch 655/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7741 - accuracy: 0.7949
    Epoch 656/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7964 - accuracy: 0.7995
    Epoch 657/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9786 - accuracy: 0.8094
    Epoch 658/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6987 - accuracy: 0.8303
    Epoch 659/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.2164 - accuracy: 0.8185
    Epoch 660/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6922 - accuracy: 0.8149
    Epoch 661/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5378 - accuracy: 0.7895
    Epoch 662/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.1921 - accuracy: 0.7632
    Epoch 663/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.2677 - accuracy: 0.8140
    Epoch 664/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5334 - accuracy: 0.8230
    Epoch 665/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0818 - accuracy: 0.7822
    Epoch 666/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0056 - accuracy: 0.8094
    Epoch 667/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5289 - accuracy: 0.8140
    Epoch 668/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.0279 - accuracy: 0.7695
    Epoch 669/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.9721 - accuracy: 0.7623
    Epoch 670/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5315 - accuracy: 0.8158
    Epoch 671/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9813 - accuracy: 0.7904
    Epoch 672/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7899 - accuracy: 0.7868
    Epoch 673/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4238 - accuracy: 0.7731
    Epoch 674/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7156 - accuracy: 0.8022
    Epoch 675/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2602 - accuracy: 0.8022
    Epoch 676/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.5372 - accuracy: 0.7650
    Epoch 677/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.3713 - accuracy: 0.7759
    Epoch 678/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.8571 - accuracy: 0.7659
    Epoch 679/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.4750 - accuracy: 0.7804
    Epoch 680/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.9292 - accuracy: 0.7595
    Epoch 681/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.0194 - accuracy: 0.7695
    Epoch 682/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3868 - accuracy: 0.8230
    Epoch 683/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3837 - accuracy: 0.8094
    Epoch 684/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.9059 - accuracy: 0.7677
    Epoch 685/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.8222 - accuracy: 0.7995
    Epoch 686/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.6327 - accuracy: 0.7495
    Epoch 687/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.9760 - accuracy: 0.8076
    Epoch 688/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1609 - accuracy: 0.7858
    Epoch 689/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4168 - accuracy: 0.7922
    Epoch 690/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1641 - accuracy: 0.8103
    Epoch 691/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7009 - accuracy: 0.7931
    Epoch 692/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0787 - accuracy: 0.8430
    Epoch 693/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5858 - accuracy: 0.8176
    Epoch 694/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.1510 - accuracy: 0.8421
    Epoch 695/6000
    35/35 [==============================] - 0s 559us/step - loss: 2.7622 - accuracy: 0.7868
    Epoch 696/6000
    35/35 [==============================] - 0s 470us/step - loss: 7.5056 - accuracy: 0.7495
    Epoch 697/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3576 - accuracy: 0.8303
    Epoch 698/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.8635 - accuracy: 0.7913
    Epoch 699/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.6172 - accuracy: 0.8031
    Epoch 700/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6752 - accuracy: 0.8022
    Epoch 701/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9015 - accuracy: 0.7958
    Epoch 702/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5356 - accuracy: 0.8203
    Epoch 703/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5625 - accuracy: 0.8122
    Epoch 704/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.1690 - accuracy: 0.7840
    Epoch 705/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1001 - accuracy: 0.8094
    Epoch 706/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9322 - accuracy: 0.7895
    Epoch 707/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3635 - accuracy: 0.8058
    Epoch 708/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4629 - accuracy: 0.8167
    Epoch 709/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0379 - accuracy: 0.8004
    Epoch 710/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5087 - accuracy: 0.8040
    Epoch 711/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.8970 - accuracy: 0.7677
    Epoch 712/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.9558 - accuracy: 0.7740
    Epoch 713/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7814 - accuracy: 0.8022
    Epoch 714/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7901 - accuracy: 0.8067
    Epoch 715/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.9240 - accuracy: 0.7613
    Epoch 716/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9777 - accuracy: 0.7822
    Epoch 717/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2514 - accuracy: 0.8122
    Epoch 718/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3122 - accuracy: 0.7931
    Epoch 719/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9299 - accuracy: 0.7813
    Epoch 720/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.1767 - accuracy: 0.7768
    Epoch 721/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1716 - accuracy: 0.8321
    Epoch 722/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4584 - accuracy: 0.8176
    Epoch 723/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0791 - accuracy: 0.7849
    Epoch 724/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0361 - accuracy: 0.7949
    Epoch 725/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.1777 - accuracy: 0.7759
    Epoch 726/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2222 - accuracy: 0.8040
    Epoch 727/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7918 - accuracy: 0.7985
    Epoch 728/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0958 - accuracy: 0.7895
    Epoch 729/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7638 - accuracy: 0.7922
    Epoch 730/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3716 - accuracy: 0.8031
    Epoch 731/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9649 - accuracy: 0.7913
    Epoch 732/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2593 - accuracy: 0.8013
    Epoch 733/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.0954 - accuracy: 0.7759
    Epoch 734/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7915 - accuracy: 0.7695
    Epoch 735/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5531 - accuracy: 0.7813
    Epoch 736/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1928 - accuracy: 0.8330
    Epoch 737/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0845 - accuracy: 0.8258
    Epoch 738/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1553 - accuracy: 0.8058
    Epoch 739/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5096 - accuracy: 0.8140
    Epoch 740/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.8082 - accuracy: 0.7813
    Epoch 741/6000
    35/35 [==============================] - 0s 441us/step - loss: 9.7757 - accuracy: 0.7495
    Epoch 742/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.8544 - accuracy: 0.7541
    Epoch 743/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3439 - accuracy: 0.7795
    Epoch 744/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8179 - accuracy: 0.8040
    Epoch 745/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7514 - accuracy: 0.8049
    Epoch 746/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7762 - accuracy: 0.8185
    Epoch 747/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5554 - accuracy: 0.8040
    Epoch 748/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6201 - accuracy: 0.8022
    Epoch 749/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9488 - accuracy: 0.8040
    Epoch 750/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.3824 - accuracy: 0.7650
    Epoch 751/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2388 - accuracy: 0.7995
    Epoch 752/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1508 - accuracy: 0.8285
    Epoch 753/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2518 - accuracy: 0.8330
    Epoch 754/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.2180 - accuracy: 0.8294
    Epoch 755/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.5103 - accuracy: 0.7840
    Epoch 756/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.4657 - accuracy: 0.7967
    Epoch 757/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8039 - accuracy: 0.8004
    Epoch 758/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0126 - accuracy: 0.8185
    Epoch 759/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8183 - accuracy: 0.8058
    Epoch 760/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5620 - accuracy: 0.8131
    Epoch 761/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7593 - accuracy: 0.8022
    Epoch 762/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.1351 - accuracy: 0.7695
    Epoch 763/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7021 - accuracy: 0.7650
    Epoch 764/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.6175 - accuracy: 0.7931
    Epoch 765/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2225 - accuracy: 0.7913
    Epoch 766/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7556 - accuracy: 0.8122
    Epoch 767/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6916 - accuracy: 0.7858
    Epoch 768/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3840 - accuracy: 0.8330
    Epoch 769/6000
    35/35 [==============================] - 0s 471us/step - loss: 8.5594 - accuracy: 0.7314
    Epoch 770/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3203 - accuracy: 0.7922
    Epoch 771/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4254 - accuracy: 0.8040
    Epoch 772/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.0913 - accuracy: 0.7949
    Epoch 773/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5160 - accuracy: 0.7958
    Epoch 774/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.1252 - accuracy: 0.8258
    Epoch 775/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.6306 - accuracy: 0.7813
    Epoch 776/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5990 - accuracy: 0.7750
    Epoch 777/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2248 - accuracy: 0.8403
    Epoch 778/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.0331 - accuracy: 0.7795
    Epoch 779/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.1681 - accuracy: 0.8067
    Epoch 780/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6477 - accuracy: 0.8149
    Epoch 781/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9567 - accuracy: 0.8122
    Epoch 782/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3789 - accuracy: 0.7695
    Epoch 783/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2522 - accuracy: 0.7813
    Epoch 784/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.5192 - accuracy: 0.7786
    Epoch 785/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0228 - accuracy: 0.7595
    Epoch 786/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.5242 - accuracy: 0.7750
    Epoch 787/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5173 - accuracy: 0.8067
    Epoch 788/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6699 - accuracy: 0.8149
    Epoch 789/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.9161 - accuracy: 0.7704
    Epoch 790/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.6912 - accuracy: 0.7804
    Epoch 791/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8616 - accuracy: 0.7759
    Epoch 792/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.0079 - accuracy: 0.7686
    Epoch 793/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2461 - accuracy: 0.8049
    Epoch 794/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5085 - accuracy: 0.8194
    Epoch 795/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.3495 - accuracy: 0.7795
    Epoch 796/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7446 - accuracy: 0.7958
    Epoch 797/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8714 - accuracy: 0.7913
    Epoch 798/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1508 - accuracy: 0.8085
    Epoch 799/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6641 - accuracy: 0.8049
    Epoch 800/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8167 - accuracy: 0.7840
    Epoch 801/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7947 - accuracy: 0.8113
    Epoch 802/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.5601 - accuracy: 0.7731
    Epoch 803/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8158 - accuracy: 0.8167
    Epoch 804/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8574 - accuracy: 0.8149
    Epoch 805/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1373 - accuracy: 0.7931
    Epoch 806/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3159 - accuracy: 0.8085
    Epoch 807/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2677 - accuracy: 0.8131
    Epoch 808/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5781 - accuracy: 0.7949
    Epoch 809/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2181 - accuracy: 0.8294
    Epoch 810/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5723 - accuracy: 0.7831
    Epoch 811/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3630 - accuracy: 0.7967
    Epoch 812/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5709 - accuracy: 0.8131
    Epoch 813/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3304 - accuracy: 0.7831
    Epoch 814/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6441 - accuracy: 0.8221
    Epoch 815/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.3079 - accuracy: 0.7740
    Epoch 816/6000
    35/35 [==============================] - 0s 441us/step - loss: 7.2440 - accuracy: 0.7704
    Epoch 817/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5346 - accuracy: 0.7940
    Epoch 818/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3042 - accuracy: 0.8249
    Epoch 819/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7361 - accuracy: 0.8067
    Epoch 820/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.1124 - accuracy: 0.7641
    Epoch 821/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.1687 - accuracy: 0.7922
    Epoch 822/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5841 - accuracy: 0.8122
    Epoch 823/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0554 - accuracy: 0.8167
    Epoch 824/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5595 - accuracy: 0.7904
    Epoch 825/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0911 - accuracy: 0.7895
    Epoch 826/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9957 - accuracy: 0.8221
    Epoch 827/6000
    35/35 [==============================] - 0s 471us/step - loss: 5.0112 - accuracy: 0.7686
    Epoch 828/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3456 - accuracy: 0.7822
    Epoch 829/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3673 - accuracy: 0.7731
    Epoch 830/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3510 - accuracy: 0.7895
    Epoch 831/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2798 - accuracy: 0.8067
    Epoch 832/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.8848 - accuracy: 0.7831
    Epoch 833/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8850 - accuracy: 0.7877
    Epoch 834/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6715 - accuracy: 0.8176
    Epoch 835/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5973 - accuracy: 0.7831
    Epoch 836/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9734 - accuracy: 0.8394
    Epoch 837/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0819 - accuracy: 0.7886
    Epoch 838/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2413 - accuracy: 0.7913
    Epoch 839/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3858 - accuracy: 0.7985
    Epoch 840/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.3126 - accuracy: 0.8049
    Epoch 841/6000
    35/35 [==============================] - 0s 559us/step - loss: 7.8238 - accuracy: 0.7250
    Epoch 842/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3182 - accuracy: 0.7904
    Epoch 843/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.6024 - accuracy: 0.7940
    Epoch 844/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.7314 - accuracy: 0.8094
    Epoch 845/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8703 - accuracy: 0.7940
    Epoch 846/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5319 - accuracy: 0.8149
    Epoch 847/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9908 - accuracy: 0.7895
    Epoch 848/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0202 - accuracy: 0.8004
    Epoch 849/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3917 - accuracy: 0.7677
    Epoch 850/6000
    35/35 [==============================] - 0s 471us/step - loss: 8.3039 - accuracy: 0.7405
    Epoch 851/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.2503 - accuracy: 0.7668
    Epoch 852/6000
    35/35 [==============================] - 0s 471us/step - loss: 7.1539 - accuracy: 0.7777
    Epoch 853/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.8316 - accuracy: 0.7686
    Epoch 854/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1333 - accuracy: 0.7922
    Epoch 855/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5967 - accuracy: 0.8040
    Epoch 856/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9792 - accuracy: 0.7976
    Epoch 857/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3604 - accuracy: 0.8167
    Epoch 858/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8219 - accuracy: 0.8221
    Epoch 859/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.2920 - accuracy: 0.8085
    Epoch 860/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0725 - accuracy: 0.8004
    Epoch 861/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.7837 - accuracy: 0.7877
    Epoch 862/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.3564 - accuracy: 0.7668
    Epoch 863/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.4464 - accuracy: 0.7795
    Epoch 864/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.6671 - accuracy: 0.7768
    Epoch 865/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1785 - accuracy: 0.8158
    Epoch 866/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3685 - accuracy: 0.8221
    Epoch 867/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2394 - accuracy: 0.7985
    Epoch 868/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3262 - accuracy: 0.8131
    Epoch 869/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.3061 - accuracy: 0.7985
    Epoch 870/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0281 - accuracy: 0.7958
    Epoch 871/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.9421 - accuracy: 0.8094
    Epoch 872/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9197 - accuracy: 0.8022
    Epoch 873/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2032 - accuracy: 0.8321
    Epoch 874/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5752 - accuracy: 0.7623
    Epoch 875/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1100 - accuracy: 0.8312
    Epoch 876/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8512 - accuracy: 0.8113
    Epoch 877/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4503 - accuracy: 0.8067
    Epoch 878/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9329 - accuracy: 0.8258
    Epoch 879/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1276 - accuracy: 0.8076
    Epoch 880/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.1051 - accuracy: 0.7559
    Epoch 881/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9980 - accuracy: 0.7958
    Epoch 882/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2032 - accuracy: 0.7877
    Epoch 883/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0624 - accuracy: 0.8466
    Epoch 884/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9264 - accuracy: 0.7913
    Epoch 885/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7324 - accuracy: 0.7904
    Epoch 886/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3493 - accuracy: 0.8004
    Epoch 887/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1446 - accuracy: 0.8049
    Epoch 888/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6497 - accuracy: 0.7659
    Epoch 889/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.2159 - accuracy: 0.7886
    Epoch 890/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5308 - accuracy: 0.7995
    Epoch 891/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.5719 - accuracy: 0.7840
    Epoch 892/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0236 - accuracy: 0.7713
    Epoch 893/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1934 - accuracy: 0.7813
    Epoch 894/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0357 - accuracy: 0.7623
    Epoch 895/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2820 - accuracy: 0.8094
    Epoch 896/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6796 - accuracy: 0.8076
    Epoch 897/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7961 - accuracy: 0.8049
    Epoch 898/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2513 - accuracy: 0.8131
    Epoch 899/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.6797 - accuracy: 0.7704
    Epoch 900/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2004 - accuracy: 0.7695
    Epoch 901/6000
    35/35 [==============================] - 0s 412us/step - loss: 3.0201 - accuracy: 0.7895
    Epoch 902/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3539 - accuracy: 0.8049
    Epoch 903/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8634 - accuracy: 0.8076
    Epoch 904/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8517 - accuracy: 0.8058
    Epoch 905/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.4317 - accuracy: 0.8221
    Epoch 906/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1491 - accuracy: 0.7777
    Epoch 907/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4723 - accuracy: 0.8076
    Epoch 908/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5837 - accuracy: 0.7904
    Epoch 909/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8384 - accuracy: 0.7913
    Epoch 910/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6731 - accuracy: 0.7868
    Epoch 911/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6541 - accuracy: 0.7958
    Epoch 912/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8899 - accuracy: 0.8004
    Epoch 913/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0961 - accuracy: 0.8249
    Epoch 914/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8847 - accuracy: 0.7985
    Epoch 915/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2117 - accuracy: 0.8348
    Epoch 916/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2108 - accuracy: 0.8167
    Epoch 917/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2906 - accuracy: 0.7958
    Epoch 918/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7311 - accuracy: 0.7868
    Epoch 919/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5903 - accuracy: 0.8058
    Epoch 920/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.3315 - accuracy: 0.7976
    Epoch 921/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7474 - accuracy: 0.8076
    Epoch 922/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.8572 - accuracy: 0.8103
    Epoch 923/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3272 - accuracy: 0.7995
    Epoch 924/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2646 - accuracy: 0.8276
    Epoch 925/6000
    35/35 [==============================] - 0s 441us/step - loss: 8.4430 - accuracy: 0.7668
    Epoch 926/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.9313 - accuracy: 0.7604
    Epoch 927/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1978 - accuracy: 0.8049
    Epoch 928/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2921 - accuracy: 0.8167
    Epoch 929/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7888 - accuracy: 0.7849
    Epoch 930/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.5702 - accuracy: 0.7895
    Epoch 931/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.1299 - accuracy: 0.7858
    Epoch 932/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.4827 - accuracy: 0.8194
    Epoch 933/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2484 - accuracy: 0.7967
    Epoch 934/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1950 - accuracy: 0.8103
    Epoch 935/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.2592 - accuracy: 0.7786
    Epoch 936/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8127 - accuracy: 0.8194
    Epoch 937/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7381 - accuracy: 0.7731
    Epoch 938/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7055 - accuracy: 0.7985
    Epoch 939/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6290 - accuracy: 0.8230
    Epoch 940/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5401 - accuracy: 0.7895
    Epoch 941/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5301 - accuracy: 0.8076
    Epoch 942/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9858 - accuracy: 0.7976
    Epoch 943/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3425 - accuracy: 0.7958
    Epoch 944/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.3108 - accuracy: 0.7795
    Epoch 945/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3753 - accuracy: 0.7868
    Epoch 946/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8982 - accuracy: 0.8403
    Epoch 947/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4844 - accuracy: 0.8076
    Epoch 948/6000
    35/35 [==============================] - 0s 412us/step - loss: 2.2466 - accuracy: 0.8067
    Epoch 949/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0399 - accuracy: 0.7713
    Epoch 950/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6604 - accuracy: 0.8185
    Epoch 951/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5430 - accuracy: 0.8258
    Epoch 952/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5887 - accuracy: 0.8185
    Epoch 953/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2249 - accuracy: 0.8040
    Epoch 954/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6563 - accuracy: 0.8230
    Epoch 955/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5133 - accuracy: 0.7840
    Epoch 956/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9441 - accuracy: 0.8367
    Epoch 957/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3644 - accuracy: 0.8321
    Epoch 958/6000
    35/35 [==============================] - 0s 529us/step - loss: 3.4693 - accuracy: 0.7786
    Epoch 959/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2912 - accuracy: 0.8348
    Epoch 960/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4406 - accuracy: 0.8212
    Epoch 961/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0220 - accuracy: 0.8321
    Epoch 962/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6830 - accuracy: 0.8176
    Epoch 963/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6242 - accuracy: 0.7985
    Epoch 964/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6580 - accuracy: 0.8167
    Epoch 965/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8516 - accuracy: 0.7777
    Epoch 966/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.2304 - accuracy: 0.8312
    Epoch 967/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8611 - accuracy: 0.7995
    Epoch 968/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0714 - accuracy: 0.8221
    Epoch 969/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7133 - accuracy: 0.8131
    Epoch 970/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9624 - accuracy: 0.8113
    Epoch 971/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8185 - accuracy: 0.7995
    Epoch 972/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1989 - accuracy: 0.8167
    Epoch 973/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2878 - accuracy: 0.7822
    Epoch 974/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1106 - accuracy: 0.8058
    Epoch 975/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.4740 - accuracy: 0.8076
    Epoch 976/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9611 - accuracy: 0.8013
    Epoch 977/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7400 - accuracy: 0.7877
    Epoch 978/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0823 - accuracy: 0.8085
    Epoch 979/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4686 - accuracy: 0.8031
    Epoch 980/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2049 - accuracy: 0.7922
    Epoch 981/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9275 - accuracy: 0.8058
    Epoch 982/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1660 - accuracy: 0.7877
    Epoch 983/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5363 - accuracy: 0.7913
    Epoch 984/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.2379 - accuracy: 0.7604
    Epoch 985/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5306 - accuracy: 0.7940
    Epoch 986/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.2864 - accuracy: 0.8049
    Epoch 987/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.7071 - accuracy: 0.7804
    Epoch 988/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.6442 - accuracy: 0.7849
    Epoch 989/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2999 - accuracy: 0.8022
    Epoch 990/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.9877 - accuracy: 0.7831
    Epoch 991/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0349 - accuracy: 0.8330
    Epoch 992/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5420 - accuracy: 0.8022
    Epoch 993/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.7603 - accuracy: 0.7849
    Epoch 994/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0591 - accuracy: 0.8358
    Epoch 995/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0210 - accuracy: 0.7831
    Epoch 996/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3698 - accuracy: 0.8049
    Epoch 997/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8474 - accuracy: 0.7931
    Epoch 998/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5663 - accuracy: 0.7840
    Epoch 999/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.1188 - accuracy: 0.7913
    Epoch 1000/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.0871 - accuracy: 0.7840
    Epoch 1001/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3278 - accuracy: 0.8122
    Epoch 1002/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3600 - accuracy: 0.7849
    Epoch 1003/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.6260 - accuracy: 0.7786
    Epoch 1004/6000
    35/35 [==============================] - 0s 618us/step - loss: 3.6780 - accuracy: 0.7804
    Epoch 1005/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3398 - accuracy: 0.8348
    Epoch 1006/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9364 - accuracy: 0.8276
    Epoch 1007/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3472 - accuracy: 0.7931
    Epoch 1008/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.1030 - accuracy: 0.7831
    Epoch 1009/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6092 - accuracy: 0.8040
    Epoch 1010/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5196 - accuracy: 0.7831
    Epoch 1011/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8181 - accuracy: 0.8040
    Epoch 1012/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9839 - accuracy: 0.7940
    Epoch 1013/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3456 - accuracy: 0.8267
    Epoch 1014/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4321 - accuracy: 0.7677
    Epoch 1015/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9437 - accuracy: 0.7795
    Epoch 1016/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.9080 - accuracy: 0.7686
    Epoch 1017/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8896 - accuracy: 0.8122
    Epoch 1018/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7736 - accuracy: 0.7985
    Epoch 1019/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5663 - accuracy: 0.7922
    Epoch 1020/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7369 - accuracy: 0.8103
    Epoch 1021/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2889 - accuracy: 0.8040
    Epoch 1022/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6294 - accuracy: 0.8131
    Epoch 1023/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.4968 - accuracy: 0.7886
    Epoch 1024/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1093 - accuracy: 0.8294
    Epoch 1025/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5341 - accuracy: 0.8131
    Epoch 1026/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.2363 - accuracy: 0.7740
    Epoch 1027/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.0667 - accuracy: 0.7958
    Epoch 1028/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.3308 - accuracy: 0.7940
    Epoch 1029/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.6862 - accuracy: 0.7849
    Epoch 1030/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3951 - accuracy: 0.8158
    Epoch 1031/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.4437 - accuracy: 0.7813
    Epoch 1032/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7237 - accuracy: 0.7976
    Epoch 1033/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3122 - accuracy: 0.8031
    Epoch 1034/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0655 - accuracy: 0.7822
    Epoch 1035/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.6563 - accuracy: 0.8049
    Epoch 1036/6000
    35/35 [==============================] - 0s 588us/step - loss: 2.5914 - accuracy: 0.8004
    Epoch 1037/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.8191 - accuracy: 0.7740
    Epoch 1038/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5809 - accuracy: 0.8031
    Epoch 1039/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4069 - accuracy: 0.8113
    Epoch 1040/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4657 - accuracy: 0.8031
    Epoch 1041/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2495 - accuracy: 0.7967
    Epoch 1042/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0149 - accuracy: 0.7849
    Epoch 1043/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7276 - accuracy: 0.7759
    Epoch 1044/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.2105 - accuracy: 0.7786
    Epoch 1045/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.7355 - accuracy: 0.7913
    Epoch 1046/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9499 - accuracy: 0.8385
    Epoch 1047/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1839 - accuracy: 0.8058
    Epoch 1048/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.6746 - accuracy: 0.7895
    Epoch 1049/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4940 - accuracy: 0.8094
    Epoch 1050/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6035 - accuracy: 0.8140
    Epoch 1051/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.7393 - accuracy: 0.7686
    Epoch 1052/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0203 - accuracy: 0.7759
    Epoch 1053/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0987 - accuracy: 0.8013
    Epoch 1054/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4322 - accuracy: 0.7976
    Epoch 1055/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2863 - accuracy: 0.8249
    Epoch 1056/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2117 - accuracy: 0.7913
    Epoch 1057/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.8529 - accuracy: 0.7668
    Epoch 1058/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.2807 - accuracy: 0.7913
    Epoch 1059/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2436 - accuracy: 0.8076
    Epoch 1060/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1833 - accuracy: 0.8040
    Epoch 1061/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5499 - accuracy: 0.8076
    Epoch 1062/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6391 - accuracy: 0.7967
    Epoch 1063/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4166 - accuracy: 0.8194
    Epoch 1064/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3578 - accuracy: 0.8203
    Epoch 1065/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4969 - accuracy: 0.7985
    Epoch 1066/6000
    35/35 [==============================] - 0s 470us/step - loss: 2.8081 - accuracy: 0.7731
    Epoch 1067/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2902 - accuracy: 0.7768
    Epoch 1068/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8230 - accuracy: 0.7949
    Epoch 1069/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6034 - accuracy: 0.8176
    Epoch 1070/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9642 - accuracy: 0.8285
    Epoch 1071/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8987 - accuracy: 0.8267
    Epoch 1072/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.0236 - accuracy: 0.7849
    Epoch 1073/6000
    35/35 [==============================] - 0s 559us/step - loss: 2.1439 - accuracy: 0.7949
    Epoch 1074/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.6938 - accuracy: 0.7868
    Epoch 1075/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3218 - accuracy: 0.8122
    Epoch 1076/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5391 - accuracy: 0.7967
    Epoch 1077/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5798 - accuracy: 0.8140
    Epoch 1078/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.0181 - accuracy: 0.7813
    Epoch 1079/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4255 - accuracy: 0.8004
    Epoch 1080/6000
    35/35 [==============================] - 0s 470us/step - loss: 2.1343 - accuracy: 0.7949
    Epoch 1081/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2888 - accuracy: 0.7940
    Epoch 1082/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2569 - accuracy: 0.8212
    Epoch 1083/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4958 - accuracy: 0.8212
    Epoch 1084/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7457 - accuracy: 0.8022
    Epoch 1085/6000
    35/35 [==============================] - 0s 559us/step - loss: 2.9159 - accuracy: 0.7858
    Epoch 1086/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4019 - accuracy: 0.8167
    Epoch 1087/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2999 - accuracy: 0.8167
    Epoch 1088/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5727 - accuracy: 0.7976
    Epoch 1089/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9015 - accuracy: 0.8004
    Epoch 1090/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.0476 - accuracy: 0.7877
    Epoch 1091/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.8662 - accuracy: 0.7831
    Epoch 1092/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.0827 - accuracy: 0.7405
    Epoch 1093/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.7538 - accuracy: 0.8185
    Epoch 1094/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.0982 - accuracy: 0.8194
    Epoch 1095/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9712 - accuracy: 0.8339
    Epoch 1096/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8051 - accuracy: 0.8485
    Epoch 1097/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9926 - accuracy: 0.8330
    Epoch 1098/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1116 - accuracy: 0.8212
    Epoch 1099/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9725 - accuracy: 0.7858
    Epoch 1100/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.3757 - accuracy: 0.7704
    Epoch 1101/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.2437 - accuracy: 0.7967
    Epoch 1102/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.1187 - accuracy: 0.7886
    Epoch 1103/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.0154 - accuracy: 0.8439
    Epoch 1104/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8805 - accuracy: 0.8367
    Epoch 1105/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6132 - accuracy: 0.7967
    Epoch 1106/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3204 - accuracy: 0.8203
    Epoch 1107/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2739 - accuracy: 0.8131
    Epoch 1108/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3494 - accuracy: 0.8176
    Epoch 1109/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6565 - accuracy: 0.8022
    Epoch 1110/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7809 - accuracy: 0.7613
    Epoch 1111/6000
    35/35 [==============================] - 0s 559us/step - loss: 3.2535 - accuracy: 0.7804
    Epoch 1112/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.4193 - accuracy: 0.7868
    Epoch 1113/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3048 - accuracy: 0.8194
    Epoch 1114/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2168 - accuracy: 0.8321
    Epoch 1115/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9047 - accuracy: 0.8103
    Epoch 1116/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8430 - accuracy: 0.7895
    Epoch 1117/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9947 - accuracy: 0.8439
    Epoch 1118/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0630 - accuracy: 0.8212
    Epoch 1119/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.4188 - accuracy: 0.7949
    Epoch 1120/6000
    35/35 [==============================] - 0s 529us/step - loss: 5.7832 - accuracy: 0.7722
    Epoch 1121/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1193 - accuracy: 0.8140
    Epoch 1122/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1564 - accuracy: 0.8258
    Epoch 1123/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3442 - accuracy: 0.8103
    Epoch 1124/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5880 - accuracy: 0.8167
    Epoch 1125/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0600 - accuracy: 0.7777
    Epoch 1126/6000
    35/35 [==============================] - 0s 588us/step - loss: 2.0189 - accuracy: 0.8049
    Epoch 1127/6000
    35/35 [==============================] - 0s 529us/step - loss: 3.6936 - accuracy: 0.7568
    Epoch 1128/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6104 - accuracy: 0.8085
    Epoch 1129/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.7028 - accuracy: 0.7641
    Epoch 1130/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.1460 - accuracy: 0.7695
    Epoch 1131/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9948 - accuracy: 0.7840
    Epoch 1132/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4241 - accuracy: 0.8094
    Epoch 1133/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9277 - accuracy: 0.7740
    Epoch 1134/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7806 - accuracy: 0.7840
    Epoch 1135/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2471 - accuracy: 0.7940
    Epoch 1136/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.4677 - accuracy: 0.7922
    Epoch 1137/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3367 - accuracy: 0.8013
    Epoch 1138/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2592 - accuracy: 0.8276
    Epoch 1139/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.5990 - accuracy: 0.7922
    Epoch 1140/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.9492 - accuracy: 0.7849
    Epoch 1141/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5388 - accuracy: 0.8094
    Epoch 1142/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.6167 - accuracy: 0.7840
    Epoch 1143/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2926 - accuracy: 0.8094
    Epoch 1144/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2837 - accuracy: 0.8194
    Epoch 1145/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2876 - accuracy: 0.7904
    Epoch 1146/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3728 - accuracy: 0.8267
    Epoch 1147/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4103 - accuracy: 0.8140
    Epoch 1148/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9957 - accuracy: 0.8049
    Epoch 1149/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2255 - accuracy: 0.7722
    Epoch 1150/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.6553 - accuracy: 0.7722
    Epoch 1151/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3808 - accuracy: 0.7686
    Epoch 1152/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3995 - accuracy: 0.8131
    Epoch 1153/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7950 - accuracy: 0.7895
    Epoch 1154/6000
    35/35 [==============================] - 0s 471us/step - loss: 6.3263 - accuracy: 0.7595
    Epoch 1155/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9264 - accuracy: 0.7904
    Epoch 1156/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5131 - accuracy: 0.7831
    Epoch 1157/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.5890 - accuracy: 0.7931
    Epoch 1158/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8003 - accuracy: 0.7967
    Epoch 1159/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1421 - accuracy: 0.8230
    Epoch 1160/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7499 - accuracy: 0.8167
    Epoch 1161/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.4520 - accuracy: 0.7613
    Epoch 1162/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8764 - accuracy: 0.8167
    Epoch 1163/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6334 - accuracy: 0.8040
    Epoch 1164/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0497 - accuracy: 0.7949
    Epoch 1165/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0888 - accuracy: 0.8140
    Epoch 1166/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9490 - accuracy: 0.8321
    Epoch 1167/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0771 - accuracy: 0.8276
    Epoch 1168/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5508 - accuracy: 0.7804
    Epoch 1169/6000
    35/35 [==============================] - 0s 441us/step - loss: 6.6864 - accuracy: 0.7613
    Epoch 1170/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1532 - accuracy: 0.8285
    Epoch 1171/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7883 - accuracy: 0.8058
    Epoch 1172/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2895 - accuracy: 0.7722
    Epoch 1173/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7494 - accuracy: 0.7868
    Epoch 1174/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.2124 - accuracy: 0.7759
    Epoch 1175/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1807 - accuracy: 0.8267
    Epoch 1176/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5336 - accuracy: 0.7713
    Epoch 1177/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4276 - accuracy: 0.7913
    Epoch 1178/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0472 - accuracy: 0.7840
    Epoch 1179/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6209 - accuracy: 0.8230
    Epoch 1180/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.7430 - accuracy: 0.7650
    Epoch 1181/6000
    35/35 [==============================] - 0s 471us/step - loss: 4.1639 - accuracy: 0.7886
    Epoch 1182/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7235 - accuracy: 0.8140
    Epoch 1183/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7304 - accuracy: 0.8167
    Epoch 1184/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9347 - accuracy: 0.7995
    Epoch 1185/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4730 - accuracy: 0.8040
    Epoch 1186/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1393 - accuracy: 0.8276
    Epoch 1187/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0550 - accuracy: 0.8348
    Epoch 1188/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8398 - accuracy: 0.8403
    Epoch 1189/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.5629 - accuracy: 0.7813
    Epoch 1190/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9051 - accuracy: 0.7731
    Epoch 1191/6000
    35/35 [==============================] - 0s 618us/step - loss: 2.7327 - accuracy: 0.7958
    Epoch 1192/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.0640 - accuracy: 0.7750
    Epoch 1193/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0977 - accuracy: 0.7913
    Epoch 1194/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9285 - accuracy: 0.7913
    Epoch 1195/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1479 - accuracy: 0.8031
    Epoch 1196/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6818 - accuracy: 0.8149
    Epoch 1197/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.1930 - accuracy: 0.7695
    Epoch 1198/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2626 - accuracy: 0.8176
    Epoch 1199/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8005 - accuracy: 0.8058
    Epoch 1200/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7374 - accuracy: 0.8394
    Epoch 1201/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0362 - accuracy: 0.7931
    Epoch 1202/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.5387 - accuracy: 0.8040
    Epoch 1203/6000
    35/35 [==============================] - 0s 735us/step - loss: 3.4923 - accuracy: 0.7740
    Epoch 1204/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.1194 - accuracy: 0.8194
    Epoch 1205/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6655 - accuracy: 0.8103
    Epoch 1206/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7028 - accuracy: 0.7958
    Epoch 1207/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.7682 - accuracy: 0.8049
    Epoch 1208/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0425 - accuracy: 0.8040
    Epoch 1209/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.1531 - accuracy: 0.8058
    Epoch 1210/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.6272 - accuracy: 0.7613
    Epoch 1211/6000
    35/35 [==============================] - 0s 441us/step - loss: 5.5045 - accuracy: 0.7586
    Epoch 1212/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9898 - accuracy: 0.8312
    Epoch 1213/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.9779 - accuracy: 0.8176
    Epoch 1214/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.6181 - accuracy: 0.7913
    Epoch 1215/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.0724 - accuracy: 0.7704
    Epoch 1216/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4423 - accuracy: 0.8103
    Epoch 1217/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1331 - accuracy: 0.8294
    Epoch 1218/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8491 - accuracy: 0.8457
    Epoch 1219/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7817 - accuracy: 0.8403
    Epoch 1220/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3499 - accuracy: 0.7913
    Epoch 1221/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1375 - accuracy: 0.8004
    Epoch 1222/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7451 - accuracy: 0.7976
    Epoch 1223/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.8159 - accuracy: 0.7813
    Epoch 1224/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.7733 - accuracy: 0.7958
    Epoch 1225/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.9258 - accuracy: 0.8221
    Epoch 1226/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.2362 - accuracy: 0.8076
    Epoch 1227/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0768 - accuracy: 0.8294
    Epoch 1228/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9721 - accuracy: 0.8267
    Epoch 1229/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5172 - accuracy: 0.8103
    Epoch 1230/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2535 - accuracy: 0.8185
    Epoch 1231/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1828 - accuracy: 0.8203
    Epoch 1232/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3824 - accuracy: 0.8031
    Epoch 1233/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0301 - accuracy: 0.8312
    Epoch 1234/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8103 - accuracy: 0.8394
    Epoch 1235/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.4953 - accuracy: 0.8167
    Epoch 1236/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1041 - accuracy: 0.7886
    Epoch 1237/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.4678 - accuracy: 0.7985
    Epoch 1238/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.2532 - accuracy: 0.7777
    Epoch 1239/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.7313 - accuracy: 0.7559
    Epoch 1240/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.9516 - accuracy: 0.7623
    Epoch 1241/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5818 - accuracy: 0.8094
    Epoch 1242/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.3953 - accuracy: 0.8122
    Epoch 1243/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4100 - accuracy: 0.8040
    Epoch 1244/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6651 - accuracy: 0.7985
    Epoch 1245/6000
    35/35 [==============================] - 0s 559us/step - loss: 2.1198 - accuracy: 0.7777
    Epoch 1246/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.7075 - accuracy: 0.8031
    Epoch 1247/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.5394 - accuracy: 0.8140
    Epoch 1248/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1754 - accuracy: 0.8267
    Epoch 1249/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.4722 - accuracy: 0.8040
    Epoch 1250/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9553 - accuracy: 0.8294
    Epoch 1251/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.2930 - accuracy: 0.8113
    Epoch 1252/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2706 - accuracy: 0.8185
    Epoch 1253/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7894 - accuracy: 0.8430
    Epoch 1254/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0657 - accuracy: 0.7868
    Epoch 1255/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2102 - accuracy: 0.8140
    Epoch 1256/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7509 - accuracy: 0.8376
    Epoch 1257/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.9694 - accuracy: 0.8221
    Epoch 1258/6000
    35/35 [==============================] - 0s 647us/step - loss: 8.8270 - accuracy: 0.7323
    Epoch 1259/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.8473 - accuracy: 0.7958
    Epoch 1260/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4854 - accuracy: 0.8022
    Epoch 1261/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8914 - accuracy: 0.8230
    Epoch 1262/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5359 - accuracy: 0.7958
    Epoch 1263/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.6700 - accuracy: 0.7777
    Epoch 1264/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1471 - accuracy: 0.7995
    Epoch 1265/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9750 - accuracy: 0.8330
    Epoch 1266/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7666 - accuracy: 0.8403
    Epoch 1267/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9377 - accuracy: 0.8303
    Epoch 1268/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1570 - accuracy: 0.8294
    Epoch 1269/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0145 - accuracy: 0.8221
    Epoch 1270/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4543 - accuracy: 0.8221
    Epoch 1271/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.1971 - accuracy: 0.8094
    Epoch 1272/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.0543 - accuracy: 0.8013
    Epoch 1273/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.6049 - accuracy: 0.8122
    Epoch 1274/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7218 - accuracy: 0.8122
    Epoch 1275/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6275 - accuracy: 0.8040
    Epoch 1276/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3495 - accuracy: 0.8085
    Epoch 1277/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0010 - accuracy: 0.7913
    Epoch 1278/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9328 - accuracy: 0.8240
    Epoch 1279/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8584 - accuracy: 0.8348
    Epoch 1280/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9815 - accuracy: 0.8131
    Epoch 1281/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2803 - accuracy: 0.7786
    Epoch 1282/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6446 - accuracy: 0.7895
    Epoch 1283/6000
    35/35 [==============================] - 0s 647us/step - loss: 2.2777 - accuracy: 0.7813
    Epoch 1284/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.0140 - accuracy: 0.7831
    Epoch 1285/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2686 - accuracy: 0.7750
    Epoch 1286/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.4712 - accuracy: 0.7831
    Epoch 1287/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5085 - accuracy: 0.8031
    Epoch 1288/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5972 - accuracy: 0.7886
    Epoch 1289/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1478 - accuracy: 0.8076
    Epoch 1290/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.3947 - accuracy: 0.7695
    Epoch 1291/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3275 - accuracy: 0.8221
    Epoch 1292/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2771 - accuracy: 0.7858
    Epoch 1293/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4549 - accuracy: 0.7985
    Epoch 1294/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6036 - accuracy: 0.8103
    Epoch 1295/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.9234 - accuracy: 0.8403
    Epoch 1296/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.7052 - accuracy: 0.8457
    Epoch 1297/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.2803 - accuracy: 0.8230
    Epoch 1298/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.2317 - accuracy: 0.7722
    Epoch 1299/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2659 - accuracy: 0.7695
    Epoch 1300/6000
    35/35 [==============================] - 0s 530us/step - loss: 1.8585 - accuracy: 0.8076
    Epoch 1301/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4842 - accuracy: 0.7931
    Epoch 1302/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.1122 - accuracy: 0.7904
    Epoch 1303/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1466 - accuracy: 0.8103
    Epoch 1304/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5208 - accuracy: 0.8149
    Epoch 1305/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.8232 - accuracy: 0.7831
    Epoch 1306/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.0710 - accuracy: 0.7804
    Epoch 1307/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.3819 - accuracy: 0.8067
    Epoch 1308/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.5203 - accuracy: 0.7976
    Epoch 1309/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.6737 - accuracy: 0.8022
    Epoch 1310/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3462 - accuracy: 0.7922
    Epoch 1311/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2305 - accuracy: 0.8140
    Epoch 1312/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2449 - accuracy: 0.8122
    Epoch 1313/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.3055 - accuracy: 0.8240
    Epoch 1314/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2552 - accuracy: 0.8058
    Epoch 1315/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2447 - accuracy: 0.7949
    Epoch 1316/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3473 - accuracy: 0.7840
    Epoch 1317/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6664 - accuracy: 0.8457
    Epoch 1318/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7977 - accuracy: 0.8321
    Epoch 1319/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.6538 - accuracy: 0.7822
    Epoch 1320/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.6175 - accuracy: 0.8240
    Epoch 1321/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6483 - accuracy: 0.8004
    Epoch 1322/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7637 - accuracy: 0.8421
    Epoch 1323/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9301 - accuracy: 0.8267
    Epoch 1324/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3737 - accuracy: 0.8131
    Epoch 1325/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.1734 - accuracy: 0.8103
    Epoch 1326/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1600 - accuracy: 0.8185
    Epoch 1327/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5395 - accuracy: 0.7949
    Epoch 1328/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.7430 - accuracy: 0.7704
    Epoch 1329/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0280 - accuracy: 0.7895
    Epoch 1330/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.5802 - accuracy: 0.7868
    Epoch 1331/6000
    35/35 [==============================] - 0s 529us/step - loss: 6.8127 - accuracy: 0.7541
    Epoch 1332/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.8691 - accuracy: 0.7949
    Epoch 1333/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0564 - accuracy: 0.8240
    Epoch 1334/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5715 - accuracy: 0.8167
    Epoch 1335/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3877 - accuracy: 0.7949
    Epoch 1336/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3551 - accuracy: 0.8058
    Epoch 1337/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8385 - accuracy: 0.7686
    Epoch 1338/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.5334 - accuracy: 0.8140
    Epoch 1339/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8659 - accuracy: 0.8185
    Epoch 1340/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2473 - accuracy: 0.8031
    Epoch 1341/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.7497 - accuracy: 0.7904
    Epoch 1342/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.2305 - accuracy: 0.8240
    Epoch 1343/6000
    35/35 [==============================] - 0s 676us/step - loss: 1.9911 - accuracy: 0.7868
    Epoch 1344/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.0109 - accuracy: 0.8067
    Epoch 1345/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.5178 - accuracy: 0.7722
    Epoch 1346/6000
    35/35 [==============================] - 0s 588us/step - loss: 4.2861 - accuracy: 0.7541
    Epoch 1347/6000
    35/35 [==============================] - 0s 588us/step - loss: 2.9471 - accuracy: 0.7904
    Epoch 1348/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.2790 - accuracy: 0.7967
    Epoch 1349/6000
    35/35 [==============================] - 0s 618us/step - loss: 2.7264 - accuracy: 0.7822
    Epoch 1350/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.9775 - accuracy: 0.8131
    Epoch 1351/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.8283 - accuracy: 0.8358
    Epoch 1352/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.6175 - accuracy: 0.7913
    Epoch 1353/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7235 - accuracy: 0.8258
    Epoch 1354/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.9569 - accuracy: 0.8203
    Epoch 1355/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.5723 - accuracy: 0.7831
    Epoch 1356/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0762 - accuracy: 0.8230
    Epoch 1357/6000
    35/35 [==============================] - 0s 559us/step - loss: 4.1961 - accuracy: 0.7713
    Epoch 1358/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.6522 - accuracy: 0.7822
    Epoch 1359/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.5121 - accuracy: 0.7441
    Epoch 1360/6000
    35/35 [==============================] - 0s 588us/step - loss: 6.4036 - accuracy: 0.7532
    Epoch 1361/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.9265 - accuracy: 0.8276
    Epoch 1362/6000
    35/35 [==============================] - 0s 735us/step - loss: 2.6307 - accuracy: 0.8022
    Epoch 1363/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2174 - accuracy: 0.8176
    Epoch 1364/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9782 - accuracy: 0.8085
    Epoch 1365/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.1756 - accuracy: 0.8013
    Epoch 1366/6000
    35/35 [==============================] - 0s 530us/step - loss: 1.8877 - accuracy: 0.8013
    Epoch 1367/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.7002 - accuracy: 0.7985
    Epoch 1368/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.9200 - accuracy: 0.8258
    Epoch 1369/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0690 - accuracy: 0.8230
    Epoch 1370/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7455 - accuracy: 0.8385
    Epoch 1371/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8010 - accuracy: 0.8303
    Epoch 1372/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6306 - accuracy: 0.8031
    Epoch 1373/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.8248 - accuracy: 0.7940
    Epoch 1374/6000
    35/35 [==============================] - 0s 677us/step - loss: 1.2611 - accuracy: 0.8267
    Epoch 1375/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.9380 - accuracy: 0.8276
    Epoch 1376/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.9787 - accuracy: 0.8339
    Epoch 1377/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3403 - accuracy: 0.8185
    Epoch 1378/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1464 - accuracy: 0.8276
    Epoch 1379/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7654 - accuracy: 0.8358
    Epoch 1380/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0399 - accuracy: 0.8276
    Epoch 1381/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.4216 - accuracy: 0.7686
    Epoch 1382/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.7846 - accuracy: 0.7632
    Epoch 1383/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7562 - accuracy: 0.8267
    Epoch 1384/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9674 - accuracy: 0.8240
    Epoch 1385/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2636 - accuracy: 0.8158
    Epoch 1386/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.3964 - accuracy: 0.7940
    Epoch 1387/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7136 - accuracy: 0.8430
    Epoch 1388/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0083 - accuracy: 0.8267
    Epoch 1389/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7867 - accuracy: 0.8285
    Epoch 1390/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0418 - accuracy: 0.8149
    Epoch 1391/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.6424 - accuracy: 0.8149
    Epoch 1392/6000
    35/35 [==============================] - 0s 559us/step - loss: 2.0417 - accuracy: 0.7913
    Epoch 1393/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2993 - accuracy: 0.8122
    Epoch 1394/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1586 - accuracy: 0.8158
    Epoch 1395/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2870 - accuracy: 0.8194
    Epoch 1396/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0826 - accuracy: 0.8094
    Epoch 1397/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4627 - accuracy: 0.8085
    Epoch 1398/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.2504 - accuracy: 0.7940
    Epoch 1399/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.6842 - accuracy: 0.8312
    Epoch 1400/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9519 - accuracy: 0.8285
    Epoch 1401/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2219 - accuracy: 0.7967
    Epoch 1402/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8707 - accuracy: 0.7940
    Epoch 1403/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0310 - accuracy: 0.7949
    Epoch 1404/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9637 - accuracy: 0.8176
    Epoch 1405/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8683 - accuracy: 0.8312
    Epoch 1406/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4991 - accuracy: 0.8049
    Epoch 1407/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9174 - accuracy: 0.8194
    Epoch 1408/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4234 - accuracy: 0.8103
    Epoch 1409/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.7192 - accuracy: 0.7759
    Epoch 1410/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.9083 - accuracy: 0.8249
    Epoch 1411/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.0169 - accuracy: 0.7740
    Epoch 1412/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7400 - accuracy: 0.7922
    Epoch 1413/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1557 - accuracy: 0.8176
    Epoch 1414/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2477 - accuracy: 0.7958
    Epoch 1415/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1191 - accuracy: 0.7695
    Epoch 1416/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9554 - accuracy: 0.8230
    Epoch 1417/6000
    35/35 [==============================] - 0s 530us/step - loss: 1.7929 - accuracy: 0.7795
    Epoch 1418/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.5194 - accuracy: 0.7613
    Epoch 1419/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.4982 - accuracy: 0.7541
    Epoch 1420/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1773 - accuracy: 0.8185
    Epoch 1421/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.6762 - accuracy: 0.8403
    Epoch 1422/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.8458 - accuracy: 0.8321
    Epoch 1423/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8738 - accuracy: 0.8312
    Epoch 1424/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.4436 - accuracy: 0.7858
    Epoch 1425/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.3687 - accuracy: 0.8031
    Epoch 1426/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7653 - accuracy: 0.8285
    Epoch 1427/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6381 - accuracy: 0.7849
    Epoch 1428/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.1688 - accuracy: 0.7858
    Epoch 1429/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.0764 - accuracy: 0.8158
    Epoch 1430/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.2810 - accuracy: 0.7967
    Epoch 1431/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9940 - accuracy: 0.8267
    Epoch 1432/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7529 - accuracy: 0.8221
    Epoch 1433/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4582 - accuracy: 0.7940
    Epoch 1434/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4472 - accuracy: 0.7967
    Epoch 1435/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8786 - accuracy: 0.8240
    Epoch 1436/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7620 - accuracy: 0.7731
    Epoch 1437/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.0795 - accuracy: 0.7985
    Epoch 1438/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.5081 - accuracy: 0.7586
    Epoch 1439/6000
    35/35 [==============================] - 0s 500us/step - loss: 4.0799 - accuracy: 0.7704
    Epoch 1440/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.9037 - accuracy: 0.8022
    Epoch 1441/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9659 - accuracy: 0.7786
    Epoch 1442/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8917 - accuracy: 0.8203
    Epoch 1443/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0007 - accuracy: 0.8094
    Epoch 1444/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0158 - accuracy: 0.8149
    Epoch 1445/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0096 - accuracy: 0.8176
    Epoch 1446/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5987 - accuracy: 0.8530
    Epoch 1447/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4495 - accuracy: 0.8294
    Epoch 1448/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.6455 - accuracy: 0.7641
    Epoch 1449/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.6732 - accuracy: 0.8167
    Epoch 1450/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8251 - accuracy: 0.7913
    Epoch 1451/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1734 - accuracy: 0.8058
    Epoch 1452/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.2917 - accuracy: 0.7840
    Epoch 1453/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.4241 - accuracy: 0.7995
    Epoch 1454/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.8037 - accuracy: 0.8240
    Epoch 1455/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6372 - accuracy: 0.8457
    Epoch 1456/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9220 - accuracy: 0.8339
    Epoch 1457/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9385 - accuracy: 0.8067
    Epoch 1458/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9310 - accuracy: 0.8203
    Epoch 1459/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6405 - accuracy: 0.8203
    Epoch 1460/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8942 - accuracy: 0.8240
    Epoch 1461/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.7532 - accuracy: 0.8285
    Epoch 1462/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.4328 - accuracy: 0.8094
    Epoch 1463/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.9402 - accuracy: 0.7822
    Epoch 1464/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9110 - accuracy: 0.8221
    Epoch 1465/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8390 - accuracy: 0.8194
    Epoch 1466/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1479 - accuracy: 0.8122
    Epoch 1467/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7689 - accuracy: 0.8376
    Epoch 1468/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.8179 - accuracy: 0.7958
    Epoch 1469/6000
    35/35 [==============================] - 0s 559us/step - loss: 2.3641 - accuracy: 0.7759
    Epoch 1470/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9696 - accuracy: 0.8221
    Epoch 1471/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.6455 - accuracy: 0.8394
    Epoch 1472/6000
    35/35 [==============================] - 0s 676us/step - loss: 2.1362 - accuracy: 0.7995
    Epoch 1473/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.4350 - accuracy: 0.7886
    Epoch 1474/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.4313 - accuracy: 0.8067
    Epoch 1475/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.9937 - accuracy: 0.7958
    Epoch 1476/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.1326 - accuracy: 0.8085
    Epoch 1477/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0496 - accuracy: 0.7922
    Epoch 1478/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.0391 - accuracy: 0.8113
    Epoch 1479/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.6235 - accuracy: 0.8466
    Epoch 1480/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.5765 - accuracy: 0.7686
    Epoch 1481/6000
    35/35 [==============================] - 0s 618us/step - loss: 3.5332 - accuracy: 0.7813
    Epoch 1482/6000
    35/35 [==============================] - 0s 588us/step - loss: 3.3027 - accuracy: 0.7677
    Epoch 1483/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.3773 - accuracy: 0.7750
    Epoch 1484/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.9037 - accuracy: 0.7795
    Epoch 1485/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0606 - accuracy: 0.8031
    Epoch 1486/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.0049 - accuracy: 0.7604
    Epoch 1487/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2390 - accuracy: 0.7931
    Epoch 1488/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8620 - accuracy: 0.8249
    Epoch 1489/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.8494 - accuracy: 0.8285
    Epoch 1490/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8675 - accuracy: 0.7831
    Epoch 1491/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.8670 - accuracy: 0.7659
    Epoch 1492/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0648 - accuracy: 0.7895
    Epoch 1493/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.4068 - accuracy: 0.7886
    Epoch 1494/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.3809 - accuracy: 0.8113
    Epoch 1495/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0780 - accuracy: 0.8203
    Epoch 1496/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4552 - accuracy: 0.7931
    Epoch 1497/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9702 - accuracy: 0.8040
    Epoch 1498/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7050 - accuracy: 0.8494
    Epoch 1499/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2829 - accuracy: 0.7958
    Epoch 1500/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3907 - accuracy: 0.7886
    Epoch 1501/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.7054 - accuracy: 0.7849
    Epoch 1502/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1003 - accuracy: 0.8031
    Epoch 1503/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4345 - accuracy: 0.7895
    Epoch 1504/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.6886 - accuracy: 0.8094
    Epoch 1505/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.1498 - accuracy: 0.8094
    Epoch 1506/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0089 - accuracy: 0.8176
    Epoch 1507/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8802 - accuracy: 0.8158
    Epoch 1508/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6583 - accuracy: 0.8421
    Epoch 1509/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2110 - accuracy: 0.8067
    Epoch 1510/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6663 - accuracy: 0.7886
    Epoch 1511/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4475 - accuracy: 0.8049
    Epoch 1512/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8255 - accuracy: 0.7895
    Epoch 1513/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3397 - accuracy: 0.8031
    Epoch 1514/6000
    35/35 [==============================] - 0s 677us/step - loss: 1.1012 - accuracy: 0.8176
    Epoch 1515/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.9352 - accuracy: 0.7795
    Epoch 1516/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.8644 - accuracy: 0.8249
    Epoch 1517/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0005 - accuracy: 0.8040
    Epoch 1518/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7608 - accuracy: 0.8376
    Epoch 1519/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0098 - accuracy: 0.8140
    Epoch 1520/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6596 - accuracy: 0.8439
    Epoch 1521/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9352 - accuracy: 0.8221
    Epoch 1522/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6306 - accuracy: 0.8412
    Epoch 1523/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6892 - accuracy: 0.8358
    Epoch 1524/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2461 - accuracy: 0.8103
    Epoch 1525/6000
    35/35 [==============================] - 0s 706us/step - loss: 0.9831 - accuracy: 0.8140
    Epoch 1526/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.2778 - accuracy: 0.7985
    Epoch 1527/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9265 - accuracy: 0.8285
    Epoch 1528/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3509 - accuracy: 0.8185
    Epoch 1529/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3523 - accuracy: 0.7704
    Epoch 1530/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4451 - accuracy: 0.8103
    Epoch 1531/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.7771 - accuracy: 0.8240
    Epoch 1532/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.7782 - accuracy: 0.8403
    Epoch 1533/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.0067 - accuracy: 0.8194
    Epoch 1534/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.8985 - accuracy: 0.7750
    Epoch 1535/6000
    35/35 [==============================] - 0s 824us/step - loss: 0.8479 - accuracy: 0.8267
    Epoch 1536/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.7121 - accuracy: 0.8367
    Epoch 1537/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1465 - accuracy: 0.8103
    Epoch 1538/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0182 - accuracy: 0.8258
    Epoch 1539/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6632 - accuracy: 0.8348
    Epoch 1540/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7348 - accuracy: 0.8303
    Epoch 1541/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9553 - accuracy: 0.8058
    Epoch 1542/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7224 - accuracy: 0.8348
    Epoch 1543/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0201 - accuracy: 0.8285
    Epoch 1544/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8089 - accuracy: 0.8185
    Epoch 1545/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1639 - accuracy: 0.8131
    Epoch 1546/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.9188 - accuracy: 0.8049
    Epoch 1547/6000
    35/35 [==============================] - 0s 588us/step - loss: 2.8963 - accuracy: 0.7768
    Epoch 1548/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.2539 - accuracy: 0.7849
    Epoch 1549/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8682 - accuracy: 0.8203
    Epoch 1550/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1879 - accuracy: 0.8013
    Epoch 1551/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3362 - accuracy: 0.8076
    Epoch 1552/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.0763 - accuracy: 0.7804
    Epoch 1553/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3994 - accuracy: 0.8122
    Epoch 1554/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3691 - accuracy: 0.8022
    Epoch 1555/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.9419 - accuracy: 0.8167
    Epoch 1556/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.0513 - accuracy: 0.8094
    Epoch 1557/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6536 - accuracy: 0.8485
    Epoch 1558/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7851 - accuracy: 0.8285
    Epoch 1559/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6866 - accuracy: 0.8394
    Epoch 1560/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9676 - accuracy: 0.8303
    Epoch 1561/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9441 - accuracy: 0.8240
    Epoch 1562/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.8548 - accuracy: 0.7904
    Epoch 1563/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1881 - accuracy: 0.7822
    Epoch 1564/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.8248 - accuracy: 0.8294
    Epoch 1565/6000
    35/35 [==============================] - 0s 882us/step - loss: 0.9188 - accuracy: 0.8094
    Epoch 1566/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.1929 - accuracy: 0.8058
    Epoch 1567/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.9097 - accuracy: 0.7659
    Epoch 1568/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.0159 - accuracy: 0.8185
    Epoch 1569/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3042 - accuracy: 0.8203
    Epoch 1570/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6336 - accuracy: 0.8267
    Epoch 1571/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0570 - accuracy: 0.8113
    Epoch 1572/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6611 - accuracy: 0.8176
    Epoch 1573/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9872 - accuracy: 0.8094
    Epoch 1574/6000
    35/35 [==============================] - 0s 676us/step - loss: 1.2312 - accuracy: 0.8085
    Epoch 1575/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.0122 - accuracy: 0.8058
    Epoch 1576/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4981 - accuracy: 0.8494
    Epoch 1577/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8859 - accuracy: 0.8294
    Epoch 1578/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.9709 - accuracy: 0.8013
    Epoch 1579/6000
    35/35 [==============================] - 0s 559us/step - loss: 2.1864 - accuracy: 0.7922
    Epoch 1580/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.7782 - accuracy: 0.8249
    Epoch 1581/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.8612 - accuracy: 0.7740
    Epoch 1582/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.1567 - accuracy: 0.8330
    Epoch 1583/6000
    35/35 [==============================] - 0s 912us/step - loss: 1.2138 - accuracy: 0.8103
    Epoch 1584/6000
    35/35 [==============================] - 0s 765us/step - loss: 0.8376 - accuracy: 0.8339
    Epoch 1585/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.1298 - accuracy: 0.8022
    Epoch 1586/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.2346 - accuracy: 0.7958
    Epoch 1587/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0166 - accuracy: 0.8212
    Epoch 1588/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.7387 - accuracy: 0.8312
    Epoch 1589/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.1377 - accuracy: 0.8140
    Epoch 1590/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.8183 - accuracy: 0.8230
    Epoch 1591/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7066 - accuracy: 0.8303
    Epoch 1592/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.9232 - accuracy: 0.8230
    Epoch 1593/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.5243 - accuracy: 0.7713
    Epoch 1594/6000
    35/35 [==============================] - 0s 853us/step - loss: 1.2510 - accuracy: 0.8094
    Epoch 1595/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.6054 - accuracy: 0.7886
    Epoch 1596/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8127 - accuracy: 0.8240
    Epoch 1597/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.7579 - accuracy: 0.8221
    Epoch 1598/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.0502 - accuracy: 0.7995
    Epoch 1599/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.6883 - accuracy: 0.8240
    Epoch 1600/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.2230 - accuracy: 0.7995
    Epoch 1601/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.8130 - accuracy: 0.8312
    Epoch 1602/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.5579 - accuracy: 0.8194
    Epoch 1603/6000
    35/35 [==============================] - 0s 794us/step - loss: 0.8066 - accuracy: 0.8176
    Epoch 1604/6000
    35/35 [==============================] - 0s 676us/step - loss: 1.1667 - accuracy: 0.7949
    Epoch 1605/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.5409 - accuracy: 0.7840
    Epoch 1606/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.0773 - accuracy: 0.8113
    Epoch 1607/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8506 - accuracy: 0.8258
    Epoch 1608/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.6449 - accuracy: 0.8430
    Epoch 1609/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1304 - accuracy: 0.8131
    Epoch 1610/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.4704 - accuracy: 0.7958
    Epoch 1611/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9592 - accuracy: 0.8167
    Epoch 1612/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5710 - accuracy: 0.8403
    Epoch 1613/6000
    35/35 [==============================] - 0s 853us/step - loss: 0.6109 - accuracy: 0.8412
    Epoch 1614/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.0003 - accuracy: 0.8258
    Epoch 1615/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.6082 - accuracy: 0.7877
    Epoch 1616/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.9545 - accuracy: 0.8149
    Epoch 1617/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3677 - accuracy: 0.7958
    Epoch 1618/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.7920 - accuracy: 0.8058
    Epoch 1619/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.7810 - accuracy: 0.7895
    Epoch 1620/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.7807 - accuracy: 0.8330
    Epoch 1621/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.5809 - accuracy: 0.8394
    Epoch 1622/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.7315 - accuracy: 0.8367
    Epoch 1623/6000
    35/35 [==============================] - 0s 853us/step - loss: 0.9129 - accuracy: 0.8058
    Epoch 1624/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2599 - accuracy: 0.7995
    Epoch 1625/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9170 - accuracy: 0.8140
    Epoch 1626/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.8626 - accuracy: 0.8140
    Epoch 1627/6000
    35/35 [==============================] - 0s 824us/step - loss: 1.4348 - accuracy: 0.8058
    Epoch 1628/6000
    35/35 [==============================] - 0s 647us/step - loss: 4.3141 - accuracy: 0.7568
    Epoch 1629/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.4423 - accuracy: 0.7904
    Epoch 1630/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9829 - accuracy: 0.7949
    Epoch 1631/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2594 - accuracy: 0.7985
    Epoch 1632/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.3373 - accuracy: 0.7595
    Epoch 1633/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.9333 - accuracy: 0.8058
    Epoch 1634/6000
    35/35 [==============================] - 0s 617us/step - loss: 0.7260 - accuracy: 0.8203
    Epoch 1635/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9456 - accuracy: 0.8339
    Epoch 1636/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1673 - accuracy: 0.8076
    Epoch 1637/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8474 - accuracy: 0.8185
    Epoch 1638/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5816 - accuracy: 0.8385
    Epoch 1639/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4742 - accuracy: 0.7958
    Epoch 1640/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8124 - accuracy: 0.8131
    Epoch 1641/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.5834 - accuracy: 0.7931
    Epoch 1642/6000
    35/35 [==============================] - 0s 824us/step - loss: 2.3475 - accuracy: 0.7904
    Epoch 1643/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9916 - accuracy: 0.8122
    Epoch 1644/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.0616 - accuracy: 0.8049
    Epoch 1645/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6011 - accuracy: 0.8403
    Epoch 1646/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1859 - accuracy: 0.8022
    Epoch 1647/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0774 - accuracy: 0.8103
    Epoch 1648/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6580 - accuracy: 0.8412
    Epoch 1649/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6851 - accuracy: 0.8249
    Epoch 1650/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3601 - accuracy: 0.7922
    Epoch 1651/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8239 - accuracy: 0.8149
    Epoch 1652/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.5910 - accuracy: 0.7831
    Epoch 1653/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8259 - accuracy: 0.8203
    Epoch 1654/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3446 - accuracy: 0.7922
    Epoch 1655/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7611 - accuracy: 0.8158
    Epoch 1656/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3271 - accuracy: 0.7958
    Epoch 1657/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2964 - accuracy: 0.7958
    Epoch 1658/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2892 - accuracy: 0.7976
    Epoch 1659/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0407 - accuracy: 0.8022
    Epoch 1660/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0455 - accuracy: 0.8140
    Epoch 1661/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0286 - accuracy: 0.7659
    Epoch 1662/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.7871 - accuracy: 0.8339
    Epoch 1663/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8661 - accuracy: 0.8258
    Epoch 1664/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.6388 - accuracy: 0.8321
    Epoch 1665/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.7934 - accuracy: 0.8303
    Epoch 1666/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.7350 - accuracy: 0.8158
    Epoch 1667/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.6543 - accuracy: 0.7940
    Epoch 1668/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.7039 - accuracy: 0.7813
    Epoch 1669/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.3287 - accuracy: 0.7886
    Epoch 1670/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.3927 - accuracy: 0.7840
    Epoch 1671/6000
    35/35 [==============================] - 0s 735us/step - loss: 1.0427 - accuracy: 0.8240
    Epoch 1672/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7042 - accuracy: 0.8348
    Epoch 1673/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5635 - accuracy: 0.8412
    Epoch 1674/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5338 - accuracy: 0.7904
    Epoch 1675/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6293 - accuracy: 0.8421
    Epoch 1676/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5471 - accuracy: 0.8457
    Epoch 1677/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5506 - accuracy: 0.8475
    Epoch 1678/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8200 - accuracy: 0.8185
    Epoch 1679/6000
    35/35 [==============================] - 0s 735us/step - loss: 0.7466 - accuracy: 0.8321
    Epoch 1680/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.5951 - accuracy: 0.8321
    Epoch 1681/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.0754 - accuracy: 0.7858
    Epoch 1682/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5266 - accuracy: 0.7922
    Epoch 1683/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.6325 - accuracy: 0.7922
    Epoch 1684/6000
    35/35 [==============================] - 0s 470us/step - loss: 2.5626 - accuracy: 0.7813
    Epoch 1685/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.2949 - accuracy: 0.7668
    Epoch 1686/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0848 - accuracy: 0.8131
    Epoch 1687/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.5250 - accuracy: 0.7868
    Epoch 1688/6000
    35/35 [==============================] - 0s 853us/step - loss: 1.5039 - accuracy: 0.8049
    Epoch 1689/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.1563 - accuracy: 0.8103
    Epoch 1690/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7963 - accuracy: 0.8131
    Epoch 1691/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.5932 - accuracy: 0.8312
    Epoch 1692/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.8475 - accuracy: 0.8240
    Epoch 1693/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.4621 - accuracy: 0.7895
    Epoch 1694/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.9719 - accuracy: 0.8022
    Epoch 1695/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.0163 - accuracy: 0.8203
    Epoch 1696/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.8452 - accuracy: 0.8194
    Epoch 1697/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2208 - accuracy: 0.8031
    Epoch 1698/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.8813 - accuracy: 0.8212
    Epoch 1699/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.6957 - accuracy: 0.8421
    Epoch 1700/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.6521 - accuracy: 0.8321
    Epoch 1701/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.0921 - accuracy: 0.7967
    Epoch 1702/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7916 - accuracy: 0.8158
    Epoch 1703/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9705 - accuracy: 0.8312
    Epoch 1704/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.7986 - accuracy: 0.8158
    Epoch 1705/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.0664 - accuracy: 0.8149
    Epoch 1706/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9621 - accuracy: 0.8103
    Epoch 1707/6000
    35/35 [==============================] - 0s 618us/step - loss: 1.2157 - accuracy: 0.7985
    Epoch 1708/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.7755 - accuracy: 0.8312
    Epoch 1709/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.7628 - accuracy: 0.8122
    Epoch 1710/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.7522 - accuracy: 0.8312
    Epoch 1711/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.5656 - accuracy: 0.8457
    Epoch 1712/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7263 - accuracy: 0.8221
    Epoch 1713/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.2191 - accuracy: 0.7740
    Epoch 1714/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8473 - accuracy: 0.7886
    Epoch 1715/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5532 - accuracy: 0.8421
    Epoch 1716/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6132 - accuracy: 0.7831
    Epoch 1717/6000
    35/35 [==============================] - 0s 794us/step - loss: 0.7723 - accuracy: 0.8194
    Epoch 1718/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.9440 - accuracy: 0.7985
    Epoch 1719/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8961 - accuracy: 0.8158
    Epoch 1720/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7602 - accuracy: 0.8221
    Epoch 1721/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1693 - accuracy: 0.8040
    Epoch 1722/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2081 - accuracy: 0.7886
    Epoch 1723/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.3667 - accuracy: 0.7877
    Epoch 1724/6000
    35/35 [==============================] - 0s 471us/step - loss: 3.8858 - accuracy: 0.7550
    Epoch 1725/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1879 - accuracy: 0.8103
    Epoch 1726/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.1139 - accuracy: 0.7913
    Epoch 1727/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.5089 - accuracy: 0.7895
    Epoch 1728/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9618 - accuracy: 0.8058
    Epoch 1729/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.0464 - accuracy: 0.7822
    Epoch 1730/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1400 - accuracy: 0.8004
    Epoch 1731/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7633 - accuracy: 0.8385
    Epoch 1732/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7548 - accuracy: 0.8276
    Epoch 1733/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6146 - accuracy: 0.8385
    Epoch 1734/6000
    35/35 [==============================] - 0s 706us/step - loss: 0.8852 - accuracy: 0.8167
    Epoch 1735/6000
    35/35 [==============================] - 0s 529us/step - loss: 4.4984 - accuracy: 0.7677
    Epoch 1736/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.9059 - accuracy: 0.7904
    Epoch 1737/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6058 - accuracy: 0.8276
    Epoch 1738/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9956 - accuracy: 0.7895
    Epoch 1739/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2050 - accuracy: 0.7849
    Epoch 1740/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.3054 - accuracy: 0.7604
    Epoch 1741/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9961 - accuracy: 0.7931
    Epoch 1742/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.0162 - accuracy: 0.7958
    Epoch 1743/6000
    35/35 [==============================] - 0s 823us/step - loss: 1.1173 - accuracy: 0.7985
    Epoch 1744/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.4182 - accuracy: 0.7913
    Epoch 1745/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.4712 - accuracy: 0.8049
    Epoch 1746/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8684 - accuracy: 0.8221
    Epoch 1747/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5984 - accuracy: 0.8367
    Epoch 1748/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7044 - accuracy: 0.8339
    Epoch 1749/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6106 - accuracy: 0.8367
    Epoch 1750/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5854 - accuracy: 0.8448
    Epoch 1751/6000
    35/35 [==============================] - 0s 765us/step - loss: 2.1321 - accuracy: 0.7795
    Epoch 1752/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1282 - accuracy: 0.8004
    Epoch 1753/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7308 - accuracy: 0.8176
    Epoch 1754/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8382 - accuracy: 0.8067
    Epoch 1755/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.7475 - accuracy: 0.7759
    Epoch 1756/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2932 - accuracy: 0.7995
    Epoch 1757/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0447 - accuracy: 0.8258
    Epoch 1758/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1742 - accuracy: 0.7813
    Epoch 1759/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.9191 - accuracy: 0.7813
    Epoch 1760/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.6522 - accuracy: 0.8294
    Epoch 1761/6000
    35/35 [==============================] - 0s 706us/step - loss: 1.0084 - accuracy: 0.8067
    Epoch 1762/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5596 - accuracy: 0.8367
    Epoch 1763/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7580 - accuracy: 0.8203
    Epoch 1764/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1670 - accuracy: 0.7868
    Epoch 1765/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7887 - accuracy: 0.8058
    Epoch 1766/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6517 - accuracy: 0.8076
    Epoch 1767/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7957 - accuracy: 0.8276
    Epoch 1768/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5106 - accuracy: 0.8521
    Epoch 1769/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5975 - accuracy: 0.8303
    Epoch 1770/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.6510 - accuracy: 0.8267
    Epoch 1771/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.0603 - accuracy: 0.8013
    Epoch 1772/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6796 - accuracy: 0.7695
    Epoch 1773/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5392 - accuracy: 0.8494
    Epoch 1774/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9020 - accuracy: 0.8203
    Epoch 1775/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6593 - accuracy: 0.8294
    Epoch 1776/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4351 - accuracy: 0.8648
    Epoch 1777/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6845 - accuracy: 0.8330
    Epoch 1778/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4420 - accuracy: 0.8067
    Epoch 1779/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.1303 - accuracy: 0.8094
    Epoch 1780/6000
    35/35 [==============================] - 0s 677us/step - loss: 1.2990 - accuracy: 0.7868
    Epoch 1781/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.7805 - accuracy: 0.8330
    Epoch 1782/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5972 - accuracy: 0.8330
    Epoch 1783/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7553 - accuracy: 0.8194
    Epoch 1784/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5752 - accuracy: 0.8321
    Epoch 1785/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1060 - accuracy: 0.8094
    Epoch 1786/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4816 - accuracy: 0.7731
    Epoch 1787/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8764 - accuracy: 0.8230
    Epoch 1788/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.7880 - accuracy: 0.8076
    Epoch 1789/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.7164 - accuracy: 0.8267
    Epoch 1790/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5133 - accuracy: 0.8485
    Epoch 1791/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8886 - accuracy: 0.8113
    Epoch 1792/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5945 - accuracy: 0.8348
    Epoch 1793/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7480 - accuracy: 0.8321
    Epoch 1794/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1589 - accuracy: 0.8022
    Epoch 1795/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0384 - accuracy: 0.7931
    Epoch 1796/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7170 - accuracy: 0.8212
    Epoch 1797/6000
    35/35 [==============================] - 0s 794us/step - loss: 0.7922 - accuracy: 0.8230
    Epoch 1798/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9360 - accuracy: 0.8330
    Epoch 1799/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6442 - accuracy: 0.8339
    Epoch 1800/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9454 - accuracy: 0.8094
    Epoch 1801/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6758 - accuracy: 0.8348
    Epoch 1802/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9018 - accuracy: 0.8085
    Epoch 1803/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1600 - accuracy: 0.8194
    Epoch 1804/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2275 - accuracy: 0.7868
    Epoch 1805/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3850 - accuracy: 0.7922
    Epoch 1806/6000
    35/35 [==============================] - 0s 470us/step - loss: 3.2620 - accuracy: 0.7668
    Epoch 1807/6000
    35/35 [==============================] - 0s 765us/step - loss: 3.6427 - accuracy: 0.7641
    Epoch 1808/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8713 - accuracy: 0.8240
    Epoch 1809/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0628 - accuracy: 0.7949
    Epoch 1810/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7473 - accuracy: 0.8376
    Epoch 1811/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0975 - accuracy: 0.8031
    Epoch 1812/6000
    35/35 [==============================] - 0s 500us/step - loss: 3.1185 - accuracy: 0.7523
    Epoch 1813/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1209 - accuracy: 0.8158
    Epoch 1814/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5624 - accuracy: 0.8385
    Epoch 1815/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5438 - accuracy: 0.8448
    Epoch 1816/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0618 - accuracy: 0.7976
    Epoch 1817/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.1220 - accuracy: 0.8058
    Epoch 1818/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.7533 - accuracy: 0.7895
    Epoch 1819/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0310 - accuracy: 0.7886
    Epoch 1820/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8698 - accuracy: 0.8140
    Epoch 1821/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6557 - accuracy: 0.8240
    Epoch 1822/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9244 - accuracy: 0.8040
    Epoch 1823/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5351 - accuracy: 0.8521
    Epoch 1824/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8236 - accuracy: 0.8113
    Epoch 1825/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8146 - accuracy: 0.8194
    Epoch 1826/6000
    35/35 [==============================] - 0s 706us/step - loss: 1.3322 - accuracy: 0.7985
    Epoch 1827/6000
    35/35 [==============================] - 0s 500us/step - loss: 2.3340 - accuracy: 0.7595
    Epoch 1828/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9618 - accuracy: 0.7985
    Epoch 1829/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8589 - accuracy: 0.8194
    Epoch 1830/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5318 - accuracy: 0.8312
    Epoch 1831/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8635 - accuracy: 0.8085
    Epoch 1832/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0623 - accuracy: 0.7695
    Epoch 1833/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5405 - accuracy: 0.8339
    Epoch 1834/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7577 - accuracy: 0.8158
    Epoch 1835/6000
    35/35 [==============================] - 0s 765us/step - loss: 0.6125 - accuracy: 0.8421
    Epoch 1836/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2190 - accuracy: 0.7895
    Epoch 1837/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5613 - accuracy: 0.8530
    Epoch 1838/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6914 - accuracy: 0.8367
    Epoch 1839/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8534 - accuracy: 0.8249
    Epoch 1840/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7833 - accuracy: 0.8203
    Epoch 1841/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1802 - accuracy: 0.7740
    Epoch 1842/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.2102 - accuracy: 0.7985
    Epoch 1843/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9683 - accuracy: 0.7868
    Epoch 1844/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.0207 - accuracy: 0.8122
    Epoch 1845/6000
    35/35 [==============================] - 0s 706us/step - loss: 0.9677 - accuracy: 0.8149
    Epoch 1846/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.8830 - accuracy: 0.7768
    Epoch 1847/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.1168 - accuracy: 0.8004
    Epoch 1848/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.9914 - accuracy: 0.8113
    Epoch 1849/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2450 - accuracy: 0.7976
    Epoch 1850/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1819 - accuracy: 0.7868
    Epoch 1851/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6902 - accuracy: 0.8240
    Epoch 1852/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4982 - accuracy: 0.8530
    Epoch 1853/6000
    35/35 [==============================] - 0s 706us/step - loss: 0.5622 - accuracy: 0.8466
    Epoch 1854/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6849 - accuracy: 0.8367
    Epoch 1855/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5314 - accuracy: 0.8421
    Epoch 1856/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2315 - accuracy: 0.7995
    Epoch 1857/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6742 - accuracy: 0.7976
    Epoch 1858/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5874 - accuracy: 0.7895
    Epoch 1859/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.9738 - accuracy: 0.7777
    Epoch 1860/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5498 - accuracy: 0.8466
    Epoch 1861/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.8709 - accuracy: 0.8258
    Epoch 1862/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6857 - accuracy: 0.8421
    Epoch 1863/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6186 - accuracy: 0.8212
    Epoch 1864/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8330 - accuracy: 0.8140
    Epoch 1865/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0874 - accuracy: 0.7886
    Epoch 1866/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4742 - accuracy: 0.8621
    Epoch 1867/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6504 - accuracy: 0.8312
    Epoch 1868/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5572 - accuracy: 0.8303
    Epoch 1869/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1302 - accuracy: 0.7940
    Epoch 1870/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.5438 - accuracy: 0.8412
    Epoch 1871/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6826 - accuracy: 0.8376
    Epoch 1872/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4369 - accuracy: 0.8575
    Epoch 1873/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4866 - accuracy: 0.8512
    Epoch 1874/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6606 - accuracy: 0.8221
    Epoch 1875/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0202 - accuracy: 0.8004
    Epoch 1876/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.7649 - accuracy: 0.7985
    Epoch 1877/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3005 - accuracy: 0.8022
    Epoch 1878/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5927 - accuracy: 0.8421
    Epoch 1879/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.9204 - accuracy: 0.7976
    Epoch 1880/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6194 - accuracy: 0.8412
    Epoch 1881/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6412 - accuracy: 0.8240
    Epoch 1882/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0739 - accuracy: 0.7849
    Epoch 1883/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8964 - accuracy: 0.7858
    Epoch 1884/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9236 - accuracy: 0.8167
    Epoch 1885/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2065 - accuracy: 0.7913
    Epoch 1886/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5724 - accuracy: 0.8276
    Epoch 1887/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9002 - accuracy: 0.8085
    Epoch 1888/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.5599 - accuracy: 0.8358
    Epoch 1889/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.8304 - accuracy: 0.8221
    Epoch 1890/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2404 - accuracy: 0.7904
    Epoch 1891/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0002 - accuracy: 0.7795
    Epoch 1892/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6369 - accuracy: 0.7623
    Epoch 1893/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7635 - accuracy: 0.8339
    Epoch 1894/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6229 - accuracy: 0.8330
    Epoch 1895/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.4690 - accuracy: 0.8512
    Epoch 1896/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8594 - accuracy: 0.8176
    Epoch 1897/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4962 - accuracy: 0.8566
    Epoch 1898/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5347 - accuracy: 0.8367
    Epoch 1899/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7980 - accuracy: 0.8131
    Epoch 1900/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7153 - accuracy: 0.8094
    Epoch 1901/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7120 - accuracy: 0.8348
    Epoch 1902/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4693 - accuracy: 0.8485
    Epoch 1903/6000
    35/35 [==============================] - 0s 470us/step - loss: 1.2649 - accuracy: 0.7985
    Epoch 1904/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.5334 - accuracy: 0.7686
    Epoch 1905/6000
    35/35 [==============================] - 0s 618us/step - loss: 3.4726 - accuracy: 0.7623
    Epoch 1906/6000
    35/35 [==============================] - 0s 529us/step - loss: 2.2431 - accuracy: 0.7686
    Epoch 1907/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9453 - accuracy: 0.8094
    Epoch 1908/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3777 - accuracy: 0.7731
    Epoch 1909/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.0069 - accuracy: 0.8113
    Epoch 1910/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7593 - accuracy: 0.8285
    Epoch 1911/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6230 - accuracy: 0.8294
    Epoch 1912/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7739 - accuracy: 0.8022
    Epoch 1913/6000
    35/35 [==============================] - 0s 588us/step - loss: 1.5887 - accuracy: 0.7886
    Epoch 1914/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7906 - accuracy: 0.8185
    Epoch 1915/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5109 - accuracy: 0.8358
    Epoch 1916/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6527 - accuracy: 0.8267
    Epoch 1917/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8487 - accuracy: 0.8049
    Epoch 1918/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6235 - accuracy: 0.8321
    Epoch 1919/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5533 - accuracy: 0.8367
    Epoch 1920/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8212 - accuracy: 0.8348
    Epoch 1921/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.5553 - accuracy: 0.7750
    Epoch 1922/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0426 - accuracy: 0.8149
    Epoch 1923/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6065 - accuracy: 0.8240
    Epoch 1924/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3863 - accuracy: 0.7804
    Epoch 1925/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5712 - accuracy: 0.8403
    Epoch 1926/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6689 - accuracy: 0.8303
    Epoch 1927/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.3883 - accuracy: 0.7967
    Epoch 1928/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5643 - accuracy: 0.7822
    Epoch 1929/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.7131 - accuracy: 0.8348
    Epoch 1930/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6011 - accuracy: 0.8321
    Epoch 1931/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4441 - accuracy: 0.8584
    Epoch 1932/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.0053 - accuracy: 0.7877
    Epoch 1933/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3744 - accuracy: 0.8103
    Epoch 1934/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2935 - accuracy: 0.8049
    Epoch 1935/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.0420 - accuracy: 0.7704
    Epoch 1936/6000
    35/35 [==============================] - 0s 441us/step - loss: 4.3529 - accuracy: 0.7423
    Epoch 1937/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2372 - accuracy: 0.7577
    Epoch 1938/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6777 - accuracy: 0.8076
    Epoch 1939/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4565 - accuracy: 0.8521
    Epoch 1940/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.6850 - accuracy: 0.8212
    Epoch 1941/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7141 - accuracy: 0.8330
    Epoch 1942/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8058 - accuracy: 0.8276
    Epoch 1943/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6242 - accuracy: 0.8303
    Epoch 1944/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5734 - accuracy: 0.8276
    Epoch 1945/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3937 - accuracy: 0.7541
    Epoch 1946/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5767 - accuracy: 0.8412
    Epoch 1947/6000
    35/35 [==============================] - 0s 559us/step - loss: 2.2483 - accuracy: 0.7786
    Epoch 1948/6000
    35/35 [==============================] - 0s 647us/step - loss: 1.3198 - accuracy: 0.7967
    Epoch 1949/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.7225 - accuracy: 0.8085
    Epoch 1950/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.6852 - accuracy: 0.8276
    Epoch 1951/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.6941 - accuracy: 0.8094
    Epoch 1952/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.0486 - accuracy: 0.8085
    Epoch 1953/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.6508 - accuracy: 0.8276
    Epoch 1954/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.5318 - accuracy: 0.8303
    Epoch 1955/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.6351 - accuracy: 0.8348
    Epoch 1956/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8205 - accuracy: 0.8221
    Epoch 1957/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.5144 - accuracy: 0.8494
    Epoch 1958/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0025 - accuracy: 0.7695
    Epoch 1959/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8684 - accuracy: 0.7849
    Epoch 1960/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9623 - accuracy: 0.8049
    Epoch 1961/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7739 - accuracy: 0.8103
    Epoch 1962/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.6118 - accuracy: 0.8249
    Epoch 1963/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.4707 - accuracy: 0.8603
    Epoch 1964/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6857 - accuracy: 0.8285
    Epoch 1965/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6727 - accuracy: 0.8285
    Epoch 1966/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6779 - accuracy: 0.8240
    Epoch 1967/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6166 - accuracy: 0.8385
    Epoch 1968/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6388 - accuracy: 0.8203
    Epoch 1969/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4842 - accuracy: 0.8593
    Epoch 1970/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5358 - accuracy: 0.8521
    Epoch 1971/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8833 - accuracy: 0.8212
    Epoch 1972/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8967 - accuracy: 0.8113
    Epoch 1973/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5882 - accuracy: 0.8430
    Epoch 1974/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6976 - accuracy: 0.8376
    Epoch 1975/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8235 - accuracy: 0.8194
    Epoch 1976/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8189 - accuracy: 0.8167
    Epoch 1977/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.7273 - accuracy: 0.7486
    Epoch 1978/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.2498 - accuracy: 0.8094
    Epoch 1979/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5472 - accuracy: 0.8312
    Epoch 1980/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8668 - accuracy: 0.8031
    Epoch 1981/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0514 - accuracy: 0.8094
    Epoch 1982/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7620 - accuracy: 0.8176
    Epoch 1983/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5285 - accuracy: 0.8503
    Epoch 1984/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6213 - accuracy: 0.8176
    Epoch 1985/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6326 - accuracy: 0.8385
    Epoch 1986/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7183 - accuracy: 0.8185
    Epoch 1987/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9026 - accuracy: 0.7958
    Epoch 1988/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0127 - accuracy: 0.7858
    Epoch 1989/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5969 - accuracy: 0.8267
    Epoch 1990/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7800 - accuracy: 0.8221
    Epoch 1991/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9721 - accuracy: 0.7868
    Epoch 1992/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2985 - accuracy: 0.7858
    Epoch 1993/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8137 - accuracy: 0.8076
    Epoch 1994/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6979 - accuracy: 0.8240
    Epoch 1995/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7295 - accuracy: 0.8158
    Epoch 1996/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7830 - accuracy: 0.8094
    Epoch 1997/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9180 - accuracy: 0.7949
    Epoch 1998/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7763 - accuracy: 0.8140
    Epoch 1999/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0549 - accuracy: 0.7976
    Epoch 2000/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5493 - accuracy: 0.8394
    Epoch 2001/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5359 - accuracy: 0.8330
    Epoch 2002/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8320 - accuracy: 0.8167
    Epoch 2003/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4909 - accuracy: 0.7777
    Epoch 2004/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6714 - accuracy: 0.8267
    Epoch 2005/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5339 - accuracy: 0.8394
    Epoch 2006/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.8038 - accuracy: 0.8230
    Epoch 2007/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9903 - accuracy: 0.8049
    Epoch 2008/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.3932 - accuracy: 0.7750
    Epoch 2009/6000
    35/35 [==============================] - 0s 471us/step - loss: 2.0309 - accuracy: 0.7713
    Epoch 2010/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7533 - accuracy: 0.8131
    Epoch 2011/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4190 - accuracy: 0.8557
    Epoch 2012/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.1502 - accuracy: 0.7831
    Epoch 2013/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1248 - accuracy: 0.7967
    Epoch 2014/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4544 - accuracy: 0.8612
    Epoch 2015/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5213 - accuracy: 0.8439
    Epoch 2016/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0107 - accuracy: 0.8058
    Epoch 2017/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8186 - accuracy: 0.8113
    Epoch 2018/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4599 - accuracy: 0.8512
    Epoch 2019/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6878 - accuracy: 0.8258
    Epoch 2020/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8758 - accuracy: 0.8067
    Epoch 2021/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6368 - accuracy: 0.8212
    Epoch 2022/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8673 - accuracy: 0.7822
    Epoch 2023/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5509 - accuracy: 0.8412
    Epoch 2024/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3490 - accuracy: 0.7731
    Epoch 2025/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7271 - accuracy: 0.8303
    Epoch 2026/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9182 - accuracy: 0.8131
    Epoch 2027/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6432 - accuracy: 0.8376
    Epoch 2028/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9140 - accuracy: 0.8203
    Epoch 2029/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3535 - accuracy: 0.8040
    Epoch 2030/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7408 - accuracy: 0.8158
    Epoch 2031/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7463 - accuracy: 0.8312
    Epoch 2032/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4870 - accuracy: 0.8485
    Epoch 2033/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7688 - accuracy: 0.8022
    Epoch 2034/6000
    35/35 [==============================] - 0s 529us/step - loss: 1.0392 - accuracy: 0.7958
    Epoch 2035/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.7864 - accuracy: 0.8022
    Epoch 2036/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5438 - accuracy: 0.8367
    Epoch 2037/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5592 - accuracy: 0.8221
    Epoch 2038/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5140 - accuracy: 0.8521
    Epoch 2039/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5180 - accuracy: 0.8394
    Epoch 2040/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4838 - accuracy: 0.8521
    Epoch 2041/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.2699 - accuracy: 0.7759
    Epoch 2042/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7140 - accuracy: 0.8403
    Epoch 2043/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6741 - accuracy: 0.8230
    Epoch 2044/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6386 - accuracy: 0.8367
    Epoch 2045/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8748 - accuracy: 0.8094
    Epoch 2046/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8576 - accuracy: 0.8094
    Epoch 2047/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6191 - accuracy: 0.8403
    Epoch 2048/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4727 - accuracy: 0.8448
    Epoch 2049/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6683 - accuracy: 0.8113
    Epoch 2050/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9069 - accuracy: 0.7995
    Epoch 2051/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6518 - accuracy: 0.8267
    Epoch 2052/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4675 - accuracy: 0.8475
    Epoch 2053/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0429 - accuracy: 0.7659
    Epoch 2054/6000
    35/35 [==============================] - 0s 441us/step - loss: 3.9477 - accuracy: 0.7577
    Epoch 2055/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0230 - accuracy: 0.7858
    Epoch 2056/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7977 - accuracy: 0.8258
    Epoch 2057/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5006 - accuracy: 0.8339
    Epoch 2058/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6453 - accuracy: 0.7695
    Epoch 2059/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0524 - accuracy: 0.8131
    Epoch 2060/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6067 - accuracy: 0.8285
    Epoch 2061/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5218 - accuracy: 0.8385
    Epoch 2062/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5627 - accuracy: 0.8258
    Epoch 2063/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8645 - accuracy: 0.7949
    Epoch 2064/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7835 - accuracy: 0.8085
    Epoch 2065/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5568 - accuracy: 0.8312
    Epoch 2066/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5366 - accuracy: 0.8276
    Epoch 2067/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.0385 - accuracy: 0.7958
    Epoch 2068/6000
    35/35 [==============================] - 0s 559us/step - loss: 1.0164 - accuracy: 0.7831
    Epoch 2069/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1960 - accuracy: 0.7759
    Epoch 2070/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2525 - accuracy: 0.7995
    Epoch 2071/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4504 - accuracy: 0.8521
    Epoch 2072/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9196 - accuracy: 0.8031
    Epoch 2073/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6023 - accuracy: 0.8321
    Epoch 2074/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7512 - accuracy: 0.8004
    Epoch 2075/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5344 - accuracy: 0.8348
    Epoch 2076/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5086 - accuracy: 0.8330
    Epoch 2077/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6680 - accuracy: 0.8094
    Epoch 2078/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6022 - accuracy: 0.8412
    Epoch 2079/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4458 - accuracy: 0.8466
    Epoch 2080/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3713 - accuracy: 0.8031
    Epoch 2081/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7633 - accuracy: 0.8067
    Epoch 2082/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5702 - accuracy: 0.8303
    Epoch 2083/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7463 - accuracy: 0.8122
    Epoch 2084/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.0757 - accuracy: 0.7632
    Epoch 2085/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9574 - accuracy: 0.8067
    Epoch 2086/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5606 - accuracy: 0.8412
    Epoch 2087/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6325 - accuracy: 0.8176
    Epoch 2088/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0140 - accuracy: 0.7913
    Epoch 2089/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0316 - accuracy: 0.7985
    Epoch 2090/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3502 - accuracy: 0.8031
    Epoch 2091/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4987 - accuracy: 0.8403
    Epoch 2092/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.0000 - accuracy: 0.8049
    Epoch 2093/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7460 - accuracy: 0.8022
    Epoch 2094/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6502 - accuracy: 0.8131
    Epoch 2095/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6370 - accuracy: 0.8457
    Epoch 2096/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1834 - accuracy: 0.7868
    Epoch 2097/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1207 - accuracy: 0.7759
    Epoch 2098/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4822 - accuracy: 0.8430
    Epoch 2099/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4413 - accuracy: 0.8503
    Epoch 2100/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5344 - accuracy: 0.8475
    Epoch 2101/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5675 - accuracy: 0.8521
    Epoch 2102/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4616 - accuracy: 0.8512
    Epoch 2103/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.0662 - accuracy: 0.8058
    Epoch 2104/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5288 - accuracy: 0.8430
    Epoch 2105/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.3406 - accuracy: 0.7949
    Epoch 2106/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5759 - accuracy: 0.8194
    Epoch 2107/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.5027 - accuracy: 0.8376
    Epoch 2108/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.5924 - accuracy: 0.8249
    Epoch 2109/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8024 - accuracy: 0.8158
    Epoch 2110/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5962 - accuracy: 0.8185
    Epoch 2111/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6185 - accuracy: 0.8466
    Epoch 2112/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6469 - accuracy: 0.8466
    Epoch 2113/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.9471 - accuracy: 0.7768
    Epoch 2114/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8140 - accuracy: 0.8067
    Epoch 2115/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.9098 - accuracy: 0.7940
    Epoch 2116/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4413 - accuracy: 0.8475
    Epoch 2117/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9567 - accuracy: 0.8022
    Epoch 2118/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6155 - accuracy: 0.8394
    Epoch 2119/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7575 - accuracy: 0.8149
    Epoch 2120/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8459 - accuracy: 0.8176
    Epoch 2121/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5476 - accuracy: 0.8312
    Epoch 2122/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5807 - accuracy: 0.8212
    Epoch 2123/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9668 - accuracy: 0.8022
    Epoch 2124/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6526 - accuracy: 0.8140
    Epoch 2125/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6283 - accuracy: 0.8194
    Epoch 2126/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7285 - accuracy: 0.8176
    Epoch 2127/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5854 - accuracy: 0.8267
    Epoch 2128/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4132 - accuracy: 0.8566
    Epoch 2129/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9066 - accuracy: 0.8113
    Epoch 2130/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.1204 - accuracy: 0.7931
    Epoch 2131/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5268 - accuracy: 0.8503
    Epoch 2132/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5484 - accuracy: 0.8330
    Epoch 2133/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6196 - accuracy: 0.8403
    Epoch 2134/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6819 - accuracy: 0.8094
    Epoch 2135/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9264 - accuracy: 0.7967
    Epoch 2136/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6129 - accuracy: 0.8240
    Epoch 2137/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4975 - accuracy: 0.8466
    Epoch 2138/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6266 - accuracy: 0.7641
    Epoch 2139/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.2273 - accuracy: 0.7623
    Epoch 2140/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1303 - accuracy: 0.8158
    Epoch 2141/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4990 - accuracy: 0.8448
    Epoch 2142/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5746 - accuracy: 0.8376
    Epoch 2143/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4737 - accuracy: 0.8421
    Epoch 2144/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5169 - accuracy: 0.8457
    Epoch 2145/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4300 - accuracy: 0.8485
    Epoch 2146/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.6154 - accuracy: 0.8421
    Epoch 2147/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4343 - accuracy: 0.8512
    Epoch 2148/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9331 - accuracy: 0.8022
    Epoch 2149/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9844 - accuracy: 0.8076
    Epoch 2150/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4683 - accuracy: 0.8539
    Epoch 2151/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5049 - accuracy: 0.8457
    Epoch 2152/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4719 - accuracy: 0.8494
    Epoch 2153/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5057 - accuracy: 0.8475
    Epoch 2154/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5186 - accuracy: 0.8448
    Epoch 2155/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8239 - accuracy: 0.8040
    Epoch 2156/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7717 - accuracy: 0.8167
    Epoch 2157/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4641 - accuracy: 0.7713
    Epoch 2158/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4703 - accuracy: 0.7659
    Epoch 2159/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9387 - accuracy: 0.7931
    Epoch 2160/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3382 - accuracy: 0.7868
    Epoch 2161/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.5996 - accuracy: 0.7831
    Epoch 2162/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4383 - accuracy: 0.8485
    Epoch 2163/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6800 - accuracy: 0.8240
    Epoch 2164/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6158 - accuracy: 0.8158
    Epoch 2165/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.1408 - accuracy: 0.7922
    Epoch 2166/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7100 - accuracy: 0.8022
    Epoch 2167/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4697 - accuracy: 0.8575
    Epoch 2168/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5689 - accuracy: 0.8267
    Epoch 2169/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7715 - accuracy: 0.8040
    Epoch 2170/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7675 - accuracy: 0.8158
    Epoch 2171/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1675 - accuracy: 0.7886
    Epoch 2172/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0058 - accuracy: 0.8103
    Epoch 2173/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5273 - accuracy: 0.8339
    Epoch 2174/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5909 - accuracy: 0.8249
    Epoch 2175/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6824 - accuracy: 0.8113
    Epoch 2176/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5973 - accuracy: 0.8421
    Epoch 2177/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3971 - accuracy: 0.8630
    Epoch 2178/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6144 - accuracy: 0.8212
    Epoch 2179/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9132 - accuracy: 0.8103
    Epoch 2180/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5265 - accuracy: 0.8312
    Epoch 2181/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4326 - accuracy: 0.8548
    Epoch 2182/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8340 - accuracy: 0.8085
    Epoch 2183/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.8006 - accuracy: 0.7595
    Epoch 2184/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4951 - accuracy: 0.8448
    Epoch 2185/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8130 - accuracy: 0.8031
    Epoch 2186/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8134 - accuracy: 0.8094
    Epoch 2187/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4261 - accuracy: 0.8539
    Epoch 2188/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7911 - accuracy: 0.8158
    Epoch 2189/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6007 - accuracy: 0.8385
    Epoch 2190/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7150 - accuracy: 0.7995
    Epoch 2191/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8487 - accuracy: 0.8294
    Epoch 2192/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5587 - accuracy: 0.8439
    Epoch 2193/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6847 - accuracy: 0.8230
    Epoch 2194/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6659 - accuracy: 0.8085
    Epoch 2195/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5961 - accuracy: 0.8294
    Epoch 2196/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6299 - accuracy: 0.8158
    Epoch 2197/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5904 - accuracy: 0.8330
    Epoch 2198/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4692 - accuracy: 0.8503
    Epoch 2199/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5033 - accuracy: 0.8348
    Epoch 2200/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1126 - accuracy: 0.7976
    Epoch 2201/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0333 - accuracy: 0.7931
    Epoch 2202/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5743 - accuracy: 0.8258
    Epoch 2203/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5673 - accuracy: 0.8312
    Epoch 2204/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5402 - accuracy: 0.8403
    Epoch 2205/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5888 - accuracy: 0.8167
    Epoch 2206/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7298 - accuracy: 0.8158
    Epoch 2207/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5897 - accuracy: 0.8176
    Epoch 2208/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6016 - accuracy: 0.8312
    Epoch 2209/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4334 - accuracy: 0.8503
    Epoch 2210/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5004 - accuracy: 0.8412
    Epoch 2211/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8974 - accuracy: 0.7995
    Epoch 2212/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8152 - accuracy: 0.8221
    Epoch 2213/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7284 - accuracy: 0.8076
    Epoch 2214/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6905 - accuracy: 0.8339
    Epoch 2215/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4435 - accuracy: 0.8512
    Epoch 2216/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7160 - accuracy: 0.8158
    Epoch 2217/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6931 - accuracy: 0.8249
    Epoch 2218/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6236 - accuracy: 0.8176
    Epoch 2219/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5770 - accuracy: 0.8176
    Epoch 2220/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9125 - accuracy: 0.8004
    Epoch 2221/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4960 - accuracy: 0.8430
    Epoch 2222/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1358 - accuracy: 0.7904
    Epoch 2223/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7921 - accuracy: 0.8076
    Epoch 2224/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4880 - accuracy: 0.8512
    Epoch 2225/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7078 - accuracy: 0.8149
    Epoch 2226/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.4779 - accuracy: 0.7759
    Epoch 2227/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4019 - accuracy: 0.8621
    Epoch 2228/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5077 - accuracy: 0.8312
    Epoch 2229/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6002 - accuracy: 0.8194
    Epoch 2230/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8937 - accuracy: 0.7813
    Epoch 2231/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8464 - accuracy: 0.8249
    Epoch 2232/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.2675 - accuracy: 0.7831
    Epoch 2233/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8982 - accuracy: 0.7868
    Epoch 2234/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9221 - accuracy: 0.7931
    Epoch 2235/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5324 - accuracy: 0.8421
    Epoch 2236/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3874 - accuracy: 0.8684
    Epoch 2237/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4536 - accuracy: 0.8630
    Epoch 2238/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5690 - accuracy: 0.8303
    Epoch 2239/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6385 - accuracy: 0.8212
    Epoch 2240/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.8675 - accuracy: 0.8094
    Epoch 2241/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6854 - accuracy: 0.8058
    Epoch 2242/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8607 - accuracy: 0.8221
    Epoch 2243/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9020 - accuracy: 0.7849
    Epoch 2244/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.4338 - accuracy: 0.8494
    Epoch 2245/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.7071 - accuracy: 0.8285
    Epoch 2246/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.5217 - accuracy: 0.8321
    Epoch 2247/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5076 - accuracy: 0.8466
    Epoch 2248/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6599 - accuracy: 0.8294
    Epoch 2249/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5653 - accuracy: 0.8330
    Epoch 2250/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9651 - accuracy: 0.8004
    Epoch 2251/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9041 - accuracy: 0.7913
    Epoch 2252/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.6073 - accuracy: 0.7577
    Epoch 2253/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5253 - accuracy: 0.8575
    Epoch 2254/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.7520 - accuracy: 0.8031
    Epoch 2255/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5422 - accuracy: 0.8312
    Epoch 2256/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6026 - accuracy: 0.8303
    Epoch 2257/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.6202 - accuracy: 0.7650
    Epoch 2258/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.4760 - accuracy: 0.7895
    Epoch 2259/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8960 - accuracy: 0.8058
    Epoch 2260/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6923 - accuracy: 0.8230
    Epoch 2261/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6099 - accuracy: 0.8330
    Epoch 2262/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5424 - accuracy: 0.8403
    Epoch 2263/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4759 - accuracy: 0.8448
    Epoch 2264/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4186 - accuracy: 0.8648
    Epoch 2265/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5011 - accuracy: 0.8430
    Epoch 2266/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8807 - accuracy: 0.8103
    Epoch 2267/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5426 - accuracy: 0.8312
    Epoch 2268/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6635 - accuracy: 0.8185
    Epoch 2269/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8709 - accuracy: 0.8031
    Epoch 2270/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4135 - accuracy: 0.8548
    Epoch 2271/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4746 - accuracy: 0.8403
    Epoch 2272/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8946 - accuracy: 0.8031
    Epoch 2273/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6720 - accuracy: 0.8212
    Epoch 2274/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5089 - accuracy: 0.8394
    Epoch 2275/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4145 - accuracy: 0.8612
    Epoch 2276/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6323 - accuracy: 0.8203
    Epoch 2277/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.0021 - accuracy: 0.8049
    Epoch 2278/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8234 - accuracy: 0.8167
    Epoch 2279/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.9937 - accuracy: 0.7931
    Epoch 2280/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7089 - accuracy: 0.8194
    Epoch 2281/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6733 - accuracy: 0.8040
    Epoch 2282/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9535 - accuracy: 0.8212
    Epoch 2283/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5068 - accuracy: 0.8430
    Epoch 2284/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4614 - accuracy: 0.8421
    Epoch 2285/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4254 - accuracy: 0.8512
    Epoch 2286/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4562 - accuracy: 0.8466
    Epoch 2287/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5196 - accuracy: 0.8267
    Epoch 2288/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5485 - accuracy: 0.8285
    Epoch 2289/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5192 - accuracy: 0.8403
    Epoch 2290/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7237 - accuracy: 0.8131
    Epoch 2291/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8946 - accuracy: 0.8031
    Epoch 2292/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5489 - accuracy: 0.8285
    Epoch 2293/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6825 - accuracy: 0.8230
    Epoch 2294/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7760 - accuracy: 0.8013
    Epoch 2295/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4681 - accuracy: 0.8403
    Epoch 2296/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4008 - accuracy: 0.8666
    Epoch 2297/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4553 - accuracy: 0.8503
    Epoch 2298/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5217 - accuracy: 0.8339
    Epoch 2299/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4088 - accuracy: 0.8557
    Epoch 2300/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5422 - accuracy: 0.8403
    Epoch 2301/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3390 - accuracy: 0.7613
    Epoch 2302/6000
    35/35 [==============================] - 0s 441us/step - loss: 2.3863 - accuracy: 0.7604
    Epoch 2303/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2012 - accuracy: 0.7686
    Epoch 2304/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6781 - accuracy: 0.8240
    Epoch 2305/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.9923 - accuracy: 0.7858
    Epoch 2306/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4936 - accuracy: 0.8466
    Epoch 2307/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4792 - accuracy: 0.8303
    Epoch 2308/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4374 - accuracy: 0.8485
    Epoch 2309/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4888 - accuracy: 0.8448
    Epoch 2310/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5012 - accuracy: 0.8348
    Epoch 2311/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0636 - accuracy: 0.8013
    Epoch 2312/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7730 - accuracy: 0.8022
    Epoch 2313/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9489 - accuracy: 0.7922
    Epoch 2314/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4260 - accuracy: 0.8494
    Epoch 2315/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5114 - accuracy: 0.8403
    Epoch 2316/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6079 - accuracy: 0.8285
    Epoch 2317/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6620 - accuracy: 0.8230
    Epoch 2318/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4575 - accuracy: 0.8485
    Epoch 2319/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4674 - accuracy: 0.8294
    Epoch 2320/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7170 - accuracy: 0.8131
    Epoch 2321/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5896 - accuracy: 0.8185
    Epoch 2322/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6554 - accuracy: 0.8203
    Epoch 2323/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5365 - accuracy: 0.8448
    Epoch 2324/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8626 - accuracy: 0.8176
    Epoch 2325/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5517 - accuracy: 0.8330
    Epoch 2326/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4141 - accuracy: 0.8603
    Epoch 2327/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4965 - accuracy: 0.8421
    Epoch 2328/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5362 - accuracy: 0.8348
    Epoch 2329/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6023 - accuracy: 0.8376
    Epoch 2330/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8887 - accuracy: 0.7949
    Epoch 2331/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4895 - accuracy: 0.8312
    Epoch 2332/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4749 - accuracy: 0.8448
    Epoch 2333/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7053 - accuracy: 0.8094
    Epoch 2334/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7459 - accuracy: 0.8094
    Epoch 2335/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5873 - accuracy: 0.8221
    Epoch 2336/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6824 - accuracy: 0.8212
    Epoch 2337/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6373 - accuracy: 0.8249
    Epoch 2338/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4754 - accuracy: 0.8530
    Epoch 2339/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5034 - accuracy: 0.8439
    Epoch 2340/6000
    35/35 [==============================] - 0s 471us/step - loss: 1.1128 - accuracy: 0.7831
    Epoch 2341/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5185 - accuracy: 0.8221
    Epoch 2342/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7479 - accuracy: 0.8194
    Epoch 2343/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4943 - accuracy: 0.8430
    Epoch 2344/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6533 - accuracy: 0.8167
    Epoch 2345/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0872 - accuracy: 0.7704
    Epoch 2346/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7058 - accuracy: 0.8240
    Epoch 2347/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4309 - accuracy: 0.8457
    Epoch 2348/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5296 - accuracy: 0.8285
    Epoch 2349/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3965 - accuracy: 0.8485
    Epoch 2350/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4277 - accuracy: 0.8603
    Epoch 2351/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5872 - accuracy: 0.8131
    Epoch 2352/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7405 - accuracy: 0.8122
    Epoch 2353/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8135 - accuracy: 0.8221
    Epoch 2354/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9129 - accuracy: 0.7976
    Epoch 2355/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5872 - accuracy: 0.8294
    Epoch 2356/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4117 - accuracy: 0.8593
    Epoch 2357/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4602 - accuracy: 0.8412
    Epoch 2358/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4682 - accuracy: 0.8385
    Epoch 2359/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4151 - accuracy: 0.8521
    Epoch 2360/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.8181 - accuracy: 0.8067
    Epoch 2361/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5899 - accuracy: 0.8339
    Epoch 2362/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5125 - accuracy: 0.8312
    Epoch 2363/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3886 - accuracy: 0.8621
    Epoch 2364/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6204 - accuracy: 0.8212
    Epoch 2365/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4146 - accuracy: 0.8612
    Epoch 2366/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5309 - accuracy: 0.8385
    Epoch 2367/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4815 - accuracy: 0.8330
    Epoch 2368/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6775 - accuracy: 0.8240
    Epoch 2369/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4533 - accuracy: 0.8485
    Epoch 2370/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5861 - accuracy: 0.8312
    Epoch 2371/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4623 - accuracy: 0.8348
    Epoch 2372/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7668 - accuracy: 0.8067
    Epoch 2373/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6405 - accuracy: 0.8131
    Epoch 2374/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6289 - accuracy: 0.8094
    Epoch 2375/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6005 - accuracy: 0.8167
    Epoch 2376/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6349 - accuracy: 0.8158
    Epoch 2377/6000
    35/35 [==============================] - 0s 500us/step - loss: 1.6846 - accuracy: 0.7632
    Epoch 2378/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6820 - accuracy: 0.8085
    Epoch 2379/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8290 - accuracy: 0.7904
    Epoch 2380/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5676 - accuracy: 0.8303
    Epoch 2381/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4709 - accuracy: 0.8330
    Epoch 2382/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4068 - accuracy: 0.8575
    Epoch 2383/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5067 - accuracy: 0.8339
    Epoch 2384/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.9564 - accuracy: 0.7949
    Epoch 2385/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5584 - accuracy: 0.8212
    Epoch 2386/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4037 - accuracy: 0.8421
    Epoch 2387/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5502 - accuracy: 0.8330
    Epoch 2388/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4399 - accuracy: 0.8421
    Epoch 2389/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5206 - accuracy: 0.8294
    Epoch 2390/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4045 - accuracy: 0.8457
    Epoch 2391/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5102 - accuracy: 0.8348
    Epoch 2392/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.1356 - accuracy: 0.7813
    Epoch 2393/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.2651 - accuracy: 0.7813
    Epoch 2394/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7207 - accuracy: 0.8049
    Epoch 2395/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5324 - accuracy: 0.8212
    Epoch 2396/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4866 - accuracy: 0.8376
    Epoch 2397/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6911 - accuracy: 0.8185
    Epoch 2398/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6179 - accuracy: 0.8276
    Epoch 2399/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.1535 - accuracy: 0.7877
    Epoch 2400/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5133 - accuracy: 0.8276
    Epoch 2401/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6168 - accuracy: 0.8131
    Epoch 2402/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8801 - accuracy: 0.8176
    Epoch 2403/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5993 - accuracy: 0.8230
    Epoch 2404/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6445 - accuracy: 0.8230
    Epoch 2405/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4315 - accuracy: 0.8557
    Epoch 2406/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5874 - accuracy: 0.8376
    Epoch 2407/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4579 - accuracy: 0.8367
    Epoch 2408/6000
    35/35 [==============================] - 0s 412us/step - loss: 1.2133 - accuracy: 0.7786
    Epoch 2409/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3987 - accuracy: 0.8512
    Epoch 2410/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5314 - accuracy: 0.8267
    Epoch 2411/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7041 - accuracy: 0.8085
    Epoch 2412/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5882 - accuracy: 0.8348
    Epoch 2413/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5566 - accuracy: 0.8258
    Epoch 2414/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5353 - accuracy: 0.8367
    Epoch 2415/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5591 - accuracy: 0.8240
    Epoch 2416/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3871 - accuracy: 0.8557
    Epoch 2417/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4559 - accuracy: 0.8430
    Epoch 2418/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4352 - accuracy: 0.8593
    Epoch 2419/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8463 - accuracy: 0.8013
    Epoch 2420/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4816 - accuracy: 0.8367
    Epoch 2421/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3812 - accuracy: 0.8557
    Epoch 2422/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4796 - accuracy: 0.8557
    Epoch 2423/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5383 - accuracy: 0.8221
    Epoch 2424/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7137 - accuracy: 0.8221
    Epoch 2425/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4387 - accuracy: 0.8448
    Epoch 2426/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3787 - accuracy: 0.8603
    Epoch 2427/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3926 - accuracy: 0.8566
    Epoch 2428/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4384 - accuracy: 0.8521
    Epoch 2429/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5362 - accuracy: 0.8394
    Epoch 2430/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3946 - accuracy: 0.8621
    Epoch 2431/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4053 - accuracy: 0.8612
    Epoch 2432/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5589 - accuracy: 0.8303
    Epoch 2433/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6062 - accuracy: 0.8212
    Epoch 2434/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5802 - accuracy: 0.8249
    Epoch 2435/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8225 - accuracy: 0.7922
    Epoch 2436/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4365 - accuracy: 0.8466
    Epoch 2437/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9094 - accuracy: 0.7704
    Epoch 2438/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3322 - accuracy: 0.7722
    Epoch 2439/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3923 - accuracy: 0.8512
    Epoch 2440/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4911 - accuracy: 0.8321
    Epoch 2441/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4509 - accuracy: 0.8457
    Epoch 2442/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5311 - accuracy: 0.8303
    Epoch 2443/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4463 - accuracy: 0.8448
    Epoch 2444/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3974 - accuracy: 0.8539
    Epoch 2445/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4573 - accuracy: 0.8530
    Epoch 2446/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5158 - accuracy: 0.8348
    Epoch 2447/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7083 - accuracy: 0.8094
    Epoch 2448/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3987 - accuracy: 0.8566
    Epoch 2449/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5629 - accuracy: 0.8358
    Epoch 2450/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6334 - accuracy: 0.8067
    Epoch 2451/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7080 - accuracy: 0.8131
    Epoch 2452/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3911 - accuracy: 0.8639
    Epoch 2453/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4524 - accuracy: 0.8367
    Epoch 2454/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4346 - accuracy: 0.8503
    Epoch 2455/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4383 - accuracy: 0.8512
    Epoch 2456/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4751 - accuracy: 0.8339
    Epoch 2457/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5693 - accuracy: 0.8167
    Epoch 2458/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6143 - accuracy: 0.8167
    Epoch 2459/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5561 - accuracy: 0.8367
    Epoch 2460/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3621 - accuracy: 0.8702
    Epoch 2461/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4799 - accuracy: 0.8421
    Epoch 2462/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3520 - accuracy: 0.8730
    Epoch 2463/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4639 - accuracy: 0.8421
    Epoch 2464/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4309 - accuracy: 0.8512
    Epoch 2465/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3950 - accuracy: 0.8548
    Epoch 2466/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4023 - accuracy: 0.8557
    Epoch 2467/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4770 - accuracy: 0.8403
    Epoch 2468/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7028 - accuracy: 0.8103
    Epoch 2469/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5118 - accuracy: 0.8376
    Epoch 2470/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5071 - accuracy: 0.8485
    Epoch 2471/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5102 - accuracy: 0.8358
    Epoch 2472/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5417 - accuracy: 0.8249
    Epoch 2473/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4384 - accuracy: 0.8376
    Epoch 2474/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4661 - accuracy: 0.8475
    Epoch 2475/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4140 - accuracy: 0.8503
    Epoch 2476/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4983 - accuracy: 0.8385
    Epoch 2477/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8883 - accuracy: 0.7995
    Epoch 2478/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7696 - accuracy: 0.7913
    Epoch 2479/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6545 - accuracy: 0.8122
    Epoch 2480/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5652 - accuracy: 0.8185
    Epoch 2481/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3885 - accuracy: 0.8593
    Epoch 2482/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6630 - accuracy: 0.8049
    Epoch 2483/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4858 - accuracy: 0.8376
    Epoch 2484/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5455 - accuracy: 0.8276
    Epoch 2485/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6147 - accuracy: 0.8167
    Epoch 2486/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7262 - accuracy: 0.8040
    Epoch 2487/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.6495 - accuracy: 0.8040
    Epoch 2488/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4309 - accuracy: 0.8485
    Epoch 2489/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5824 - accuracy: 0.8203
    Epoch 2490/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5750 - accuracy: 0.8258
    Epoch 2491/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5067 - accuracy: 0.8321
    Epoch 2492/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4301 - accuracy: 0.8521
    Epoch 2493/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3763 - accuracy: 0.8711
    Epoch 2494/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5533 - accuracy: 0.8285
    Epoch 2495/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5113 - accuracy: 0.8348
    Epoch 2496/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8652 - accuracy: 0.7904
    Epoch 2497/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6409 - accuracy: 0.8258
    Epoch 2498/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5100 - accuracy: 0.8330
    Epoch 2499/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5606 - accuracy: 0.8285
    Epoch 2500/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4025 - accuracy: 0.8485
    Epoch 2501/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3437 - accuracy: 0.8739
    Epoch 2502/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8130 - accuracy: 0.8031
    Epoch 2503/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.3270 - accuracy: 0.7505
    Epoch 2504/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.8482 - accuracy: 0.7650
    Epoch 2505/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7427 - accuracy: 0.8113
    Epoch 2506/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3848 - accuracy: 0.8557
    Epoch 2507/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5006 - accuracy: 0.8303
    Epoch 2508/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3900 - accuracy: 0.8584
    Epoch 2509/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5869 - accuracy: 0.8140
    Epoch 2510/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4218 - accuracy: 0.8503
    Epoch 2511/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7071 - accuracy: 0.8040
    Epoch 2512/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4680 - accuracy: 0.8394
    Epoch 2513/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4172 - accuracy: 0.8485
    Epoch 2514/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.4129 - accuracy: 0.8630
    Epoch 2515/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5960 - accuracy: 0.8185
    Epoch 2516/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4728 - accuracy: 0.8394
    Epoch 2517/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4390 - accuracy: 0.8358
    Epoch 2518/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4115 - accuracy: 0.8575
    Epoch 2519/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4568 - accuracy: 0.8412
    Epoch 2520/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7588 - accuracy: 0.7995
    Epoch 2521/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5621 - accuracy: 0.8348
    Epoch 2522/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3507 - accuracy: 0.8657
    Epoch 2523/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6643 - accuracy: 0.8140
    Epoch 2524/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7779 - accuracy: 0.8013
    Epoch 2525/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8777 - accuracy: 0.8113
    Epoch 2526/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7957 - accuracy: 0.8004
    Epoch 2527/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5527 - accuracy: 0.8167
    Epoch 2528/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4964 - accuracy: 0.8312
    Epoch 2529/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4104 - accuracy: 0.8475
    Epoch 2530/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5153 - accuracy: 0.8203
    Epoch 2531/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3878 - accuracy: 0.8548
    Epoch 2532/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4466 - accuracy: 0.8267
    Epoch 2533/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5442 - accuracy: 0.8339
    Epoch 2534/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4370 - accuracy: 0.8448
    Epoch 2535/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3860 - accuracy: 0.8557
    Epoch 2536/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3638 - accuracy: 0.8657
    Epoch 2537/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3927 - accuracy: 0.8521
    Epoch 2538/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4125 - accuracy: 0.8503
    Epoch 2539/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3701 - accuracy: 0.8603
    Epoch 2540/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6381 - accuracy: 0.8049
    Epoch 2541/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5555 - accuracy: 0.8140
    Epoch 2542/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3796 - accuracy: 0.8521
    Epoch 2543/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4938 - accuracy: 0.8303
    Epoch 2544/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5105 - accuracy: 0.8457
    Epoch 2545/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4551 - accuracy: 0.8367
    Epoch 2546/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6358 - accuracy: 0.7895
    Epoch 2547/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8113 - accuracy: 0.7858
    Epoch 2548/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4357 - accuracy: 0.8448
    Epoch 2549/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6844 - accuracy: 0.8103
    Epoch 2550/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5049 - accuracy: 0.8321
    Epoch 2551/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7018 - accuracy: 0.8076
    Epoch 2552/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3432 - accuracy: 0.8711
    Epoch 2553/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5483 - accuracy: 0.8294
    Epoch 2554/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4932 - accuracy: 0.8439
    Epoch 2555/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4181 - accuracy: 0.8339
    Epoch 2556/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4816 - accuracy: 0.8321
    Epoch 2557/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4187 - accuracy: 0.8521
    Epoch 2558/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4609 - accuracy: 0.8312
    Epoch 2559/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3530 - accuracy: 0.8693
    Epoch 2560/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4087 - accuracy: 0.8485
    Epoch 2561/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4882 - accuracy: 0.8348
    Epoch 2562/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.8455 - accuracy: 0.7831
    Epoch 2563/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4946 - accuracy: 0.8394
    Epoch 2564/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4390 - accuracy: 0.8485
    Epoch 2565/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4034 - accuracy: 0.8485
    Epoch 2566/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3900 - accuracy: 0.8548
    Epoch 2567/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3821 - accuracy: 0.8630
    Epoch 2568/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4159 - accuracy: 0.8421
    Epoch 2569/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3706 - accuracy: 0.8612
    Epoch 2570/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3507 - accuracy: 0.8621
    Epoch 2571/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6067 - accuracy: 0.8031
    Epoch 2572/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5841 - accuracy: 0.8249
    Epoch 2573/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5331 - accuracy: 0.8358
    Epoch 2574/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3758 - accuracy: 0.8675
    Epoch 2575/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6223 - accuracy: 0.8076
    Epoch 2576/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4642 - accuracy: 0.8394
    Epoch 2577/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5742 - accuracy: 0.8185
    Epoch 2578/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6878 - accuracy: 0.8058
    Epoch 2579/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4254 - accuracy: 0.8330
    Epoch 2580/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3313 - accuracy: 0.8739
    Epoch 2581/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3713 - accuracy: 0.8566
    Epoch 2582/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3912 - accuracy: 0.8503
    Epoch 2583/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.6272 - accuracy: 0.8076
    Epoch 2584/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4007 - accuracy: 0.8566
    Epoch 2585/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6062 - accuracy: 0.8158
    Epoch 2586/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4519 - accuracy: 0.8376
    Epoch 2587/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4402 - accuracy: 0.8439
    Epoch 2588/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7731 - accuracy: 0.8267
    Epoch 2589/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5632 - accuracy: 0.8122
    Epoch 2590/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4100 - accuracy: 0.8439
    Epoch 2591/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9319 - accuracy: 0.8013
    Epoch 2592/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6969 - accuracy: 0.7886
    Epoch 2593/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7876 - accuracy: 0.7976
    Epoch 2594/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4620 - accuracy: 0.8367
    Epoch 2595/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3979 - accuracy: 0.8539
    Epoch 2596/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4918 - accuracy: 0.8221
    Epoch 2597/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3839 - accuracy: 0.8530
    Epoch 2598/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3666 - accuracy: 0.8621
    Epoch 2599/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4136 - accuracy: 0.8584
    Epoch 2600/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9041 - accuracy: 0.7831
    Epoch 2601/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5474 - accuracy: 0.8212
    Epoch 2602/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5550 - accuracy: 0.8258
    Epoch 2603/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5389 - accuracy: 0.8149
    Epoch 2604/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4150 - accuracy: 0.8385
    Epoch 2605/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4172 - accuracy: 0.8421
    Epoch 2606/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4474 - accuracy: 0.8503
    Epoch 2607/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6793 - accuracy: 0.7886
    Epoch 2608/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6031 - accuracy: 0.8176
    Epoch 2609/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3899 - accuracy: 0.8457
    Epoch 2610/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5092 - accuracy: 0.8249
    Epoch 2611/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5052 - accuracy: 0.8267
    Epoch 2612/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.7260 - accuracy: 0.7813
    Epoch 2613/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3972 - accuracy: 0.8494
    Epoch 2614/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3432 - accuracy: 0.8748
    Epoch 2615/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6304 - accuracy: 0.8122
    Epoch 2616/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6635 - accuracy: 0.7985
    Epoch 2617/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4194 - accuracy: 0.8503
    Epoch 2618/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3897 - accuracy: 0.8593
    Epoch 2619/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5559 - accuracy: 0.8276
    Epoch 2620/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5887 - accuracy: 0.8131
    Epoch 2621/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4301 - accuracy: 0.8457
    Epoch 2622/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3749 - accuracy: 0.8557
    Epoch 2623/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4915 - accuracy: 0.8312
    Epoch 2624/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4203 - accuracy: 0.8521
    Epoch 2625/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3754 - accuracy: 0.8639
    Epoch 2626/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4387 - accuracy: 0.8412
    Epoch 2627/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3805 - accuracy: 0.8575
    Epoch 2628/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3664 - accuracy: 0.8630
    Epoch 2629/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6487 - accuracy: 0.8212
    Epoch 2630/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4717 - accuracy: 0.8276
    Epoch 2631/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4482 - accuracy: 0.8367
    Epoch 2632/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3655 - accuracy: 0.8603
    Epoch 2633/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3938 - accuracy: 0.8539
    Epoch 2634/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4513 - accuracy: 0.8403
    Epoch 2635/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3828 - accuracy: 0.8566
    Epoch 2636/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4070 - accuracy: 0.8612
    Epoch 2637/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4072 - accuracy: 0.8412
    Epoch 2638/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3900 - accuracy: 0.8575
    Epoch 2639/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3574 - accuracy: 0.8548
    Epoch 2640/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3759 - accuracy: 0.8630
    Epoch 2641/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5145 - accuracy: 0.8230
    Epoch 2642/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5200 - accuracy: 0.8339
    Epoch 2643/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3779 - accuracy: 0.8548
    Epoch 2644/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3814 - accuracy: 0.8593
    Epoch 2645/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3827 - accuracy: 0.8621
    Epoch 2646/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3760 - accuracy: 0.8693
    Epoch 2647/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3708 - accuracy: 0.8693
    Epoch 2648/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4751 - accuracy: 0.8330
    Epoch 2649/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5923 - accuracy: 0.8094
    Epoch 2650/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5110 - accuracy: 0.8276
    Epoch 2651/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3375 - accuracy: 0.8702
    Epoch 2652/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4727 - accuracy: 0.8394
    Epoch 2653/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3817 - accuracy: 0.8503
    Epoch 2654/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3870 - accuracy: 0.8566
    Epoch 2655/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3836 - accuracy: 0.8548
    Epoch 2656/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3593 - accuracy: 0.8684
    Epoch 2657/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3543 - accuracy: 0.8666
    Epoch 2658/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3469 - accuracy: 0.8702
    Epoch 2659/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3704 - accuracy: 0.8603
    Epoch 2660/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5306 - accuracy: 0.8376
    Epoch 2661/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5171 - accuracy: 0.8140
    Epoch 2662/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4489 - accuracy: 0.8457
    Epoch 2663/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4409 - accuracy: 0.8385
    Epoch 2664/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4702 - accuracy: 0.8230
    Epoch 2665/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4423 - accuracy: 0.8475
    Epoch 2666/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4130 - accuracy: 0.8494
    Epoch 2667/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3860 - accuracy: 0.8466
    Epoch 2668/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5007 - accuracy: 0.8348
    Epoch 2669/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4310 - accuracy: 0.8403
    Epoch 2670/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3541 - accuracy: 0.8693
    Epoch 2671/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5014 - accuracy: 0.8421
    Epoch 2672/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4448 - accuracy: 0.8448
    Epoch 2673/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4831 - accuracy: 0.8339
    Epoch 2674/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3925 - accuracy: 0.8548
    Epoch 2675/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4540 - accuracy: 0.8475
    Epoch 2676/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7458 - accuracy: 0.8076
    Epoch 2677/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4758 - accuracy: 0.8321
    Epoch 2678/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5068 - accuracy: 0.8185
    Epoch 2679/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3894 - accuracy: 0.8448
    Epoch 2680/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4258 - accuracy: 0.8593
    Epoch 2681/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3998 - accuracy: 0.8475
    Epoch 2682/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3716 - accuracy: 0.8621
    Epoch 2683/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6234 - accuracy: 0.7967
    Epoch 2684/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5164 - accuracy: 0.8203
    Epoch 2685/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4178 - accuracy: 0.8521
    Epoch 2686/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5453 - accuracy: 0.8212
    Epoch 2687/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6565 - accuracy: 0.8348
    Epoch 2688/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5143 - accuracy: 0.8140
    Epoch 2689/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6899 - accuracy: 0.8131
    Epoch 2690/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4703 - accuracy: 0.8176
    Epoch 2691/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3745 - accuracy: 0.8557
    Epoch 2692/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4459 - accuracy: 0.8385
    Epoch 2693/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4490 - accuracy: 0.8385
    Epoch 2694/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3716 - accuracy: 0.8503
    Epoch 2695/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5682 - accuracy: 0.8131
    Epoch 2696/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3526 - accuracy: 0.8684
    Epoch 2697/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4666 - accuracy: 0.8385
    Epoch 2698/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3617 - accuracy: 0.8603
    Epoch 2699/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4421 - accuracy: 0.8385
    Epoch 2700/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6394 - accuracy: 0.8113
    Epoch 2701/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3854 - accuracy: 0.8575
    Epoch 2702/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3940 - accuracy: 0.8448
    Epoch 2703/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3795 - accuracy: 0.8521
    Epoch 2704/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4279 - accuracy: 0.8439
    Epoch 2705/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5219 - accuracy: 0.8312
    Epoch 2706/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3506 - accuracy: 0.8730
    Epoch 2707/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3732 - accuracy: 0.8512
    Epoch 2708/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6448 - accuracy: 0.8176
    Epoch 2709/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5115 - accuracy: 0.8321
    Epoch 2710/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4291 - accuracy: 0.8494
    Epoch 2711/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6446 - accuracy: 0.7995
    Epoch 2712/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4790 - accuracy: 0.8258
    Epoch 2713/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4070 - accuracy: 0.8503
    Epoch 2714/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3582 - accuracy: 0.8648
    Epoch 2715/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3532 - accuracy: 0.8548
    Epoch 2716/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3457 - accuracy: 0.8648
    Epoch 2717/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5384 - accuracy: 0.8158
    Epoch 2718/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4635 - accuracy: 0.8485
    Epoch 2719/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5050 - accuracy: 0.8267
    Epoch 2720/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4152 - accuracy: 0.8448
    Epoch 2721/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5000 - accuracy: 0.8348
    Epoch 2722/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4054 - accuracy: 0.8475
    Epoch 2723/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3783 - accuracy: 0.8584
    Epoch 2724/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3712 - accuracy: 0.8530
    Epoch 2725/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4421 - accuracy: 0.8385
    Epoch 2726/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5788 - accuracy: 0.8013
    Epoch 2727/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3949 - accuracy: 0.8503
    Epoch 2728/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.7014 - accuracy: 0.8085
    Epoch 2729/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5163 - accuracy: 0.8267
    Epoch 2730/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5760 - accuracy: 0.8258
    Epoch 2731/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5907 - accuracy: 0.8140
    Epoch 2732/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4685 - accuracy: 0.8430
    Epoch 2733/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4667 - accuracy: 0.8303
    Epoch 2734/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4314 - accuracy: 0.8485
    Epoch 2735/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4844 - accuracy: 0.8330
    Epoch 2736/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4239 - accuracy: 0.8285
    Epoch 2737/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5123 - accuracy: 0.8122
    Epoch 2738/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4050 - accuracy: 0.8466
    Epoch 2739/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4628 - accuracy: 0.8339
    Epoch 2740/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4401 - accuracy: 0.8466
    Epoch 2741/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3616 - accuracy: 0.8739
    Epoch 2742/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3784 - accuracy: 0.8593
    Epoch 2743/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3915 - accuracy: 0.8548
    Epoch 2744/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3538 - accuracy: 0.8711
    Epoch 2745/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4216 - accuracy: 0.8466
    Epoch 2746/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4026 - accuracy: 0.8376
    Epoch 2747/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3715 - accuracy: 0.8666
    Epoch 2748/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4363 - accuracy: 0.8330
    Epoch 2749/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3449 - accuracy: 0.8666
    Epoch 2750/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4181 - accuracy: 0.8457
    Epoch 2751/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3766 - accuracy: 0.8539
    Epoch 2752/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4322 - accuracy: 0.8276
    Epoch 2753/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4159 - accuracy: 0.8512
    Epoch 2754/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4301 - accuracy: 0.8403
    Epoch 2755/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4008 - accuracy: 0.8466
    Epoch 2756/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3882 - accuracy: 0.8439
    Epoch 2757/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3832 - accuracy: 0.8630
    Epoch 2758/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3627 - accuracy: 0.8548
    Epoch 2759/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3727 - accuracy: 0.8639
    Epoch 2760/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3628 - accuracy: 0.8521
    Epoch 2761/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3435 - accuracy: 0.8748
    Epoch 2762/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3842 - accuracy: 0.8512
    Epoch 2763/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4280 - accuracy: 0.8548
    Epoch 2764/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3515 - accuracy: 0.8721
    Epoch 2765/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4085 - accuracy: 0.8439
    Epoch 2766/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3790 - accuracy: 0.8584
    Epoch 2767/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5029 - accuracy: 0.8276
    Epoch 2768/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4082 - accuracy: 0.8512
    Epoch 2769/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4058 - accuracy: 0.8485
    Epoch 2770/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3770 - accuracy: 0.8566
    Epoch 2771/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4541 - accuracy: 0.8330
    Epoch 2772/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5571 - accuracy: 0.8203
    Epoch 2773/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3984 - accuracy: 0.8639
    Epoch 2774/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3412 - accuracy: 0.8666
    Epoch 2775/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3912 - accuracy: 0.8430
    Epoch 2776/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5108 - accuracy: 0.8294
    Epoch 2777/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4123 - accuracy: 0.8539
    Epoch 2778/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4404 - accuracy: 0.8385
    Epoch 2779/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5269 - accuracy: 0.8249
    Epoch 2780/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3399 - accuracy: 0.8693
    Epoch 2781/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3459 - accuracy: 0.8748
    Epoch 2782/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4253 - accuracy: 0.8394
    Epoch 2783/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3967 - accuracy: 0.8512
    Epoch 2784/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4812 - accuracy: 0.8412
    Epoch 2785/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3913 - accuracy: 0.8494
    Epoch 2786/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4540 - accuracy: 0.8312
    Epoch 2787/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6717 - accuracy: 0.7985
    Epoch 2788/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4512 - accuracy: 0.8330
    Epoch 2789/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3468 - accuracy: 0.8684
    Epoch 2790/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4255 - accuracy: 0.8466
    Epoch 2791/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5510 - accuracy: 0.8185
    Epoch 2792/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4862 - accuracy: 0.8194
    Epoch 2793/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4225 - accuracy: 0.8475
    Epoch 2794/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5111 - accuracy: 0.8167
    Epoch 2795/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4294 - accuracy: 0.8448
    Epoch 2796/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3458 - accuracy: 0.8793
    Epoch 2797/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3435 - accuracy: 0.8721
    Epoch 2798/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4591 - accuracy: 0.8339
    Epoch 2799/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5264 - accuracy: 0.8475
    Epoch 2800/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3523 - accuracy: 0.8639
    Epoch 2801/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3563 - accuracy: 0.8675
    Epoch 2802/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3485 - accuracy: 0.8702
    Epoch 2803/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4233 - accuracy: 0.8494
    Epoch 2804/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5924 - accuracy: 0.8240
    Epoch 2805/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3455 - accuracy: 0.8630
    Epoch 2806/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4485 - accuracy: 0.8312
    Epoch 2807/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4603 - accuracy: 0.8294
    Epoch 2808/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4214 - accuracy: 0.8521
    Epoch 2809/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3638 - accuracy: 0.8684
    Epoch 2810/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3983 - accuracy: 0.8521
    Epoch 2811/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4059 - accuracy: 0.8485
    Epoch 2812/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4545 - accuracy: 0.8376
    Epoch 2813/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4136 - accuracy: 0.8512
    Epoch 2814/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3860 - accuracy: 0.8557
    Epoch 2815/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3605 - accuracy: 0.8603
    Epoch 2816/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3366 - accuracy: 0.8748
    Epoch 2817/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3738 - accuracy: 0.8539
    Epoch 2818/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4941 - accuracy: 0.8122
    Epoch 2819/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3871 - accuracy: 0.8584
    Epoch 2820/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3585 - accuracy: 0.8593
    Epoch 2821/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3846 - accuracy: 0.8539
    Epoch 2822/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4252 - accuracy: 0.8485
    Epoch 2823/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3589 - accuracy: 0.8575
    Epoch 2824/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3971 - accuracy: 0.8466
    Epoch 2825/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4621 - accuracy: 0.8348
    Epoch 2826/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3473 - accuracy: 0.8684
    Epoch 2827/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5190 - accuracy: 0.8185
    Epoch 2828/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4538 - accuracy: 0.8348
    Epoch 2829/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4041 - accuracy: 0.8512
    Epoch 2830/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4153 - accuracy: 0.8421
    Epoch 2831/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3759 - accuracy: 0.8548
    Epoch 2832/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3403 - accuracy: 0.8739
    Epoch 2833/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3737 - accuracy: 0.8548
    Epoch 2834/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3576 - accuracy: 0.8603
    Epoch 2835/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3954 - accuracy: 0.8503
    Epoch 2836/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4019 - accuracy: 0.8457
    Epoch 2837/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3517 - accuracy: 0.8603
    Epoch 2838/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3949 - accuracy: 0.8448
    Epoch 2839/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4165 - accuracy: 0.8430
    Epoch 2840/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3874 - accuracy: 0.8593
    Epoch 2841/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3707 - accuracy: 0.8612
    Epoch 2842/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4270 - accuracy: 0.8412
    Epoch 2843/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3706 - accuracy: 0.8548
    Epoch 2844/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4514 - accuracy: 0.8376
    Epoch 2845/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4287 - accuracy: 0.8421
    Epoch 2846/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3831 - accuracy: 0.8503
    Epoch 2847/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3643 - accuracy: 0.8557
    Epoch 2848/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3843 - accuracy: 0.8575
    Epoch 2849/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3494 - accuracy: 0.8684
    Epoch 2850/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5813 - accuracy: 0.8131
    Epoch 2851/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4887 - accuracy: 0.8348
    Epoch 2852/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4276 - accuracy: 0.8394
    Epoch 2853/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3442 - accuracy: 0.8621
    Epoch 2854/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4003 - accuracy: 0.8494
    Epoch 2855/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5180 - accuracy: 0.8339
    Epoch 2856/6000
    35/35 [==============================] - 0s 441us/step - loss: 1.0960 - accuracy: 0.7713
    Epoch 2857/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4224 - accuracy: 0.8503
    Epoch 2858/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4462 - accuracy: 0.8466
    Epoch 2859/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4156 - accuracy: 0.8521
    Epoch 2860/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3638 - accuracy: 0.8684
    Epoch 2861/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3878 - accuracy: 0.8494
    Epoch 2862/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3966 - accuracy: 0.8475
    Epoch 2863/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4961 - accuracy: 0.8276
    Epoch 2864/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3695 - accuracy: 0.8666
    Epoch 2865/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6362 - accuracy: 0.8022
    Epoch 2866/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4117 - accuracy: 0.8485
    Epoch 2867/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3860 - accuracy: 0.8494
    Epoch 2868/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3592 - accuracy: 0.8621
    Epoch 2869/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4514 - accuracy: 0.8312
    Epoch 2870/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3635 - accuracy: 0.8648
    Epoch 2871/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3963 - accuracy: 0.8548
    Epoch 2872/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4275 - accuracy: 0.8421
    Epoch 2873/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4381 - accuracy: 0.8330
    Epoch 2874/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.6350 - accuracy: 0.8058
    Epoch 2875/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4167 - accuracy: 0.8439
    Epoch 2876/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4904 - accuracy: 0.8439
    Epoch 2877/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4241 - accuracy: 0.8448
    Epoch 2878/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3968 - accuracy: 0.8439
    Epoch 2879/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3644 - accuracy: 0.8711
    Epoch 2880/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3751 - accuracy: 0.8603
    Epoch 2881/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4118 - accuracy: 0.8521
    Epoch 2882/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4167 - accuracy: 0.8376
    Epoch 2883/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4544 - accuracy: 0.8294
    Epoch 2884/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4106 - accuracy: 0.8457
    Epoch 2885/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3757 - accuracy: 0.8612
    Epoch 2886/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4576 - accuracy: 0.8303
    Epoch 2887/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4692 - accuracy: 0.8294
    Epoch 2888/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3403 - accuracy: 0.8702
    Epoch 2889/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3865 - accuracy: 0.8566
    Epoch 2890/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6465 - accuracy: 0.8212
    Epoch 2891/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4777 - accuracy: 0.8321
    Epoch 2892/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4037 - accuracy: 0.8503
    Epoch 2893/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5117 - accuracy: 0.8267
    Epoch 2894/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4940 - accuracy: 0.8303
    Epoch 2895/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3672 - accuracy: 0.8593
    Epoch 2896/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3639 - accuracy: 0.8539
    Epoch 2897/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4132 - accuracy: 0.8466
    Epoch 2898/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3846 - accuracy: 0.8475
    Epoch 2899/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3539 - accuracy: 0.8721
    Epoch 2900/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3482 - accuracy: 0.8657
    Epoch 2901/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5553 - accuracy: 0.8203
    Epoch 2902/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3659 - accuracy: 0.8693
    Epoch 2903/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3505 - accuracy: 0.8721
    Epoch 2904/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3830 - accuracy: 0.8566
    Epoch 2905/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3691 - accuracy: 0.8566
    Epoch 2906/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4386 - accuracy: 0.8412
    Epoch 2907/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4829 - accuracy: 0.8230
    Epoch 2908/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3974 - accuracy: 0.8475
    Epoch 2909/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4173 - accuracy: 0.8485
    Epoch 2910/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4669 - accuracy: 0.8376
    Epoch 2911/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5237 - accuracy: 0.8240
    Epoch 2912/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.9038 - accuracy: 0.8212
    Epoch 2913/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4463 - accuracy: 0.8276
    Epoch 2914/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3730 - accuracy: 0.8503
    Epoch 2915/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3284 - accuracy: 0.8702
    Epoch 2916/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3448 - accuracy: 0.8684
    Epoch 2917/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4820 - accuracy: 0.8230
    Epoch 2918/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3600 - accuracy: 0.8702
    Epoch 2919/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4729 - accuracy: 0.8285
    Epoch 2920/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3498 - accuracy: 0.8793
    Epoch 2921/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3544 - accuracy: 0.8702
    Epoch 2922/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3658 - accuracy: 0.8557
    Epoch 2923/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4003 - accuracy: 0.8548
    Epoch 2924/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4539 - accuracy: 0.8394
    Epoch 2925/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3384 - accuracy: 0.8702
    Epoch 2926/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3829 - accuracy: 0.8466
    Epoch 2927/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4731 - accuracy: 0.8230
    Epoch 2928/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3765 - accuracy: 0.8566
    Epoch 2929/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4748 - accuracy: 0.8330
    Epoch 2930/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4445 - accuracy: 0.8321
    Epoch 2931/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4799 - accuracy: 0.8348
    Epoch 2932/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4568 - accuracy: 0.8203
    Epoch 2933/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3513 - accuracy: 0.8675
    Epoch 2934/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4124 - accuracy: 0.8430
    Epoch 2935/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4162 - accuracy: 0.8385
    Epoch 2936/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4910 - accuracy: 0.8330
    Epoch 2937/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3904 - accuracy: 0.8512
    Epoch 2938/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3666 - accuracy: 0.8566
    Epoch 2939/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3742 - accuracy: 0.8639
    Epoch 2940/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4302 - accuracy: 0.8475
    Epoch 2941/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4001 - accuracy: 0.8439
    Epoch 2942/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3977 - accuracy: 0.8494
    Epoch 2943/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4049 - accuracy: 0.8494
    Epoch 2944/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3609 - accuracy: 0.8512
    Epoch 2945/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4269 - accuracy: 0.8394
    Epoch 2946/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4077 - accuracy: 0.8457
    Epoch 2947/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3706 - accuracy: 0.8485
    Epoch 2948/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3499 - accuracy: 0.8639
    Epoch 2949/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3672 - accuracy: 0.8630
    Epoch 2950/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4085 - accuracy: 0.8494
    Epoch 2951/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4607 - accuracy: 0.8376
    Epoch 2952/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3399 - accuracy: 0.8666
    Epoch 2953/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3729 - accuracy: 0.8575
    Epoch 2954/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4867 - accuracy: 0.8294
    Epoch 2955/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4416 - accuracy: 0.8358
    Epoch 2956/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3765 - accuracy: 0.8639
    Epoch 2957/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3476 - accuracy: 0.8593
    Epoch 2958/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3859 - accuracy: 0.8403
    Epoch 2959/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3523 - accuracy: 0.8621
    Epoch 2960/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.5060 - accuracy: 0.8348
    Epoch 2961/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.7198 - accuracy: 0.8049
    Epoch 2962/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5344 - accuracy: 0.8285
    Epoch 2963/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3553 - accuracy: 0.8639
    Epoch 2964/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4226 - accuracy: 0.8457
    Epoch 2965/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4893 - accuracy: 0.8312
    Epoch 2966/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3911 - accuracy: 0.8603
    Epoch 2967/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3652 - accuracy: 0.8621
    Epoch 2968/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4051 - accuracy: 0.8466
    Epoch 2969/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4860 - accuracy: 0.8258
    Epoch 2970/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3601 - accuracy: 0.8702
    Epoch 2971/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3825 - accuracy: 0.8521
    Epoch 2972/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4388 - accuracy: 0.8539
    Epoch 2973/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3765 - accuracy: 0.8575
    Epoch 2974/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3313 - accuracy: 0.8784
    Epoch 2975/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4626 - accuracy: 0.8466
    Epoch 2976/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.4129 - accuracy: 0.8312
    Epoch 2977/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4333 - accuracy: 0.8521
    Epoch 2978/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3998 - accuracy: 0.8412
    Epoch 2979/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4711 - accuracy: 0.8240
    Epoch 2980/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3536 - accuracy: 0.8593
    Epoch 2981/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4727 - accuracy: 0.8276
    Epoch 2982/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4224 - accuracy: 0.8530
    Epoch 2983/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4525 - accuracy: 0.8158
    Epoch 2984/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5498 - accuracy: 0.8267
    Epoch 2985/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5488 - accuracy: 0.8140
    Epoch 2986/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4252 - accuracy: 0.8466
    Epoch 2987/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3660 - accuracy: 0.8593
    Epoch 2988/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4794 - accuracy: 0.8358
    Epoch 2989/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3452 - accuracy: 0.8648
    Epoch 2990/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3646 - accuracy: 0.8612
    Epoch 2991/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3658 - accuracy: 0.8593
    Epoch 2992/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3757 - accuracy: 0.8593
    Epoch 2993/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3731 - accuracy: 0.8603
    Epoch 2994/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3788 - accuracy: 0.8548
    Epoch 2995/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3772 - accuracy: 0.8666
    Epoch 2996/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3428 - accuracy: 0.8793
    Epoch 2997/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3499 - accuracy: 0.8675
    Epoch 2998/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3397 - accuracy: 0.8666
    Epoch 2999/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3856 - accuracy: 0.8512
    Epoch 3000/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4262 - accuracy: 0.8358
    Epoch 3001/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6225 - accuracy: 0.8113
    Epoch 3002/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4843 - accuracy: 0.8330
    Epoch 3003/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4394 - accuracy: 0.8376
    Epoch 3004/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3861 - accuracy: 0.8521
    Epoch 3005/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4112 - accuracy: 0.8548
    Epoch 3006/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8766
    Epoch 3007/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3645 - accuracy: 0.8630
    Epoch 3008/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4043 - accuracy: 0.8439
    Epoch 3009/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.4178 - accuracy: 0.8358
    Epoch 3010/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3540 - accuracy: 0.8684
    Epoch 3011/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4184 - accuracy: 0.8575
    Epoch 3012/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4638 - accuracy: 0.8167
    Epoch 3013/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4587 - accuracy: 0.8348
    Epoch 3014/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3430 - accuracy: 0.8684
    Epoch 3015/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4119 - accuracy: 0.8439
    Epoch 3016/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3627 - accuracy: 0.8675
    Epoch 3017/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3783 - accuracy: 0.8603
    Epoch 3018/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3767 - accuracy: 0.8557
    Epoch 3019/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3716 - accuracy: 0.8575
    Epoch 3020/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3617 - accuracy: 0.8639
    Epoch 3021/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3506 - accuracy: 0.8593
    Epoch 3022/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3416 - accuracy: 0.8730
    Epoch 3023/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3459 - accuracy: 0.8702
    Epoch 3024/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3437 - accuracy: 0.8711
    Epoch 3025/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4089 - accuracy: 0.8521
    Epoch 3026/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3826 - accuracy: 0.8503
    Epoch 3027/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5327 - accuracy: 0.8448
    Epoch 3028/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3822 - accuracy: 0.8485
    Epoch 3029/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3712 - accuracy: 0.8557
    Epoch 3030/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3727 - accuracy: 0.8575
    Epoch 3031/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3852 - accuracy: 0.8530
    Epoch 3032/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3528 - accuracy: 0.8639
    Epoch 3033/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3730 - accuracy: 0.8539
    Epoch 3034/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4974 - accuracy: 0.8249
    Epoch 3035/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3429 - accuracy: 0.8702
    Epoch 3036/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3730 - accuracy: 0.8612
    Epoch 3037/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4470 - accuracy: 0.8321
    Epoch 3038/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3448 - accuracy: 0.8702
    Epoch 3039/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3352 - accuracy: 0.8666
    Epoch 3040/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4081 - accuracy: 0.8367
    Epoch 3041/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3570 - accuracy: 0.8593
    Epoch 3042/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4410 - accuracy: 0.8321
    Epoch 3043/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3931 - accuracy: 0.8475
    Epoch 3044/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3396 - accuracy: 0.8675
    Epoch 3045/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3725 - accuracy: 0.8566
    Epoch 3046/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3993 - accuracy: 0.8503
    Epoch 3047/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3481 - accuracy: 0.8730
    Epoch 3048/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3868 - accuracy: 0.8494
    Epoch 3049/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3517 - accuracy: 0.8666
    Epoch 3050/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3662 - accuracy: 0.8639
    Epoch 3051/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4687 - accuracy: 0.8212
    Epoch 3052/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3801 - accuracy: 0.8557
    Epoch 3053/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5186 - accuracy: 0.8212
    Epoch 3054/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3511 - accuracy: 0.8648
    Epoch 3055/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3347 - accuracy: 0.8730
    Epoch 3056/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3619 - accuracy: 0.8584
    Epoch 3057/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4010 - accuracy: 0.8494
    Epoch 3058/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3992 - accuracy: 0.8521
    Epoch 3059/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4229 - accuracy: 0.8358
    Epoch 3060/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3467 - accuracy: 0.8657
    Epoch 3061/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3686 - accuracy: 0.8603
    Epoch 3062/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3803 - accuracy: 0.8593
    Epoch 3063/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4861 - accuracy: 0.8249
    Epoch 3064/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4427 - accuracy: 0.8448
    Epoch 3065/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3686 - accuracy: 0.8621
    Epoch 3066/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.5157 - accuracy: 0.8303
    Epoch 3067/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3722 - accuracy: 0.8503
    Epoch 3068/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3721 - accuracy: 0.8584
    Epoch 3069/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3713 - accuracy: 0.8584
    Epoch 3070/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5756 - accuracy: 0.8212
    Epoch 3071/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5425 - accuracy: 0.8131
    Epoch 3072/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3788 - accuracy: 0.8639
    Epoch 3073/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3419 - accuracy: 0.8593
    Epoch 3074/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5239 - accuracy: 0.8258
    Epoch 3075/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3600 - accuracy: 0.8766
    Epoch 3076/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3805 - accuracy: 0.8584
    Epoch 3077/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4242 - accuracy: 0.8466
    Epoch 3078/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3774 - accuracy: 0.8621
    Epoch 3079/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3734 - accuracy: 0.8612
    Epoch 3080/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4237 - accuracy: 0.8321
    Epoch 3081/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4273 - accuracy: 0.8385
    Epoch 3082/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3547 - accuracy: 0.8621
    Epoch 3083/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4130 - accuracy: 0.8494
    Epoch 3084/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3391 - accuracy: 0.8684
    Epoch 3085/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3494 - accuracy: 0.8612
    Epoch 3086/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3777 - accuracy: 0.8593
    Epoch 3087/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3755 - accuracy: 0.8621
    Epoch 3088/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4416 - accuracy: 0.8358
    Epoch 3089/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3585 - accuracy: 0.8575
    Epoch 3090/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3378 - accuracy: 0.8675
    Epoch 3091/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3503 - accuracy: 0.8603
    Epoch 3092/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5707 - accuracy: 0.8358
    Epoch 3093/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5057 - accuracy: 0.8113
    Epoch 3094/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3968 - accuracy: 0.8412
    Epoch 3095/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4115 - accuracy: 0.8439
    Epoch 3096/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4598 - accuracy: 0.8358
    Epoch 3097/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3671 - accuracy: 0.8593
    Epoch 3098/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3976 - accuracy: 0.8612
    Epoch 3099/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4520 - accuracy: 0.8339
    Epoch 3100/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4698 - accuracy: 0.8185
    Epoch 3101/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4368 - accuracy: 0.8412
    Epoch 3102/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3881 - accuracy: 0.8557
    Epoch 3103/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3521 - accuracy: 0.8748
    Epoch 3104/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3689 - accuracy: 0.8612
    Epoch 3105/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3990 - accuracy: 0.8503
    Epoch 3106/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3754 - accuracy: 0.8512
    Epoch 3107/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3633 - accuracy: 0.8521
    Epoch 3108/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3609 - accuracy: 0.8639
    Epoch 3109/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3868 - accuracy: 0.8548
    Epoch 3110/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3620 - accuracy: 0.8666
    Epoch 3111/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3864 - accuracy: 0.8584
    Epoch 3112/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3532 - accuracy: 0.8548
    Epoch 3113/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3433 - accuracy: 0.8666
    Epoch 3114/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3387 - accuracy: 0.8630
    Epoch 3115/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3426 - accuracy: 0.8603
    Epoch 3116/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3884 - accuracy: 0.8503
    Epoch 3117/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4555 - accuracy: 0.8276
    Epoch 3118/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4400 - accuracy: 0.8294
    Epoch 3119/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3463 - accuracy: 0.8702
    Epoch 3120/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4007 - accuracy: 0.8512
    Epoch 3121/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3563 - accuracy: 0.8566
    Epoch 3122/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3321 - accuracy: 0.8793
    Epoch 3123/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3828 - accuracy: 0.8530
    Epoch 3124/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3677 - accuracy: 0.8539
    Epoch 3125/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3448 - accuracy: 0.8684
    Epoch 3126/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3350 - accuracy: 0.8721
    Epoch 3127/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3491 - accuracy: 0.8575
    Epoch 3128/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3676 - accuracy: 0.8539
    Epoch 3129/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3453 - accuracy: 0.8721
    Epoch 3130/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.6105 - accuracy: 0.7931
    Epoch 3131/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4439 - accuracy: 0.8376
    Epoch 3132/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3542 - accuracy: 0.8630
    Epoch 3133/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3583 - accuracy: 0.8575
    Epoch 3134/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3886 - accuracy: 0.8521
    Epoch 3135/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3669 - accuracy: 0.8575
    Epoch 3136/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3921 - accuracy: 0.8539
    Epoch 3137/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3533 - accuracy: 0.8557
    Epoch 3138/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4207 - accuracy: 0.8457
    Epoch 3139/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4025 - accuracy: 0.8503
    Epoch 3140/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3611 - accuracy: 0.8612
    Epoch 3141/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3772 - accuracy: 0.8630
    Epoch 3142/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4956 - accuracy: 0.8122
    Epoch 3143/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3913 - accuracy: 0.8539
    Epoch 3144/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8748
    Epoch 3145/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3701 - accuracy: 0.8575
    Epoch 3146/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4398 - accuracy: 0.8394
    Epoch 3147/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4000 - accuracy: 0.8530
    Epoch 3148/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3816 - accuracy: 0.8466
    Epoch 3149/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3464 - accuracy: 0.8675
    Epoch 3150/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3496 - accuracy: 0.8648
    Epoch 3151/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3807 - accuracy: 0.8530
    Epoch 3152/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3409 - accuracy: 0.8730
    Epoch 3153/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3540 - accuracy: 0.8621
    Epoch 3154/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3767 - accuracy: 0.8566
    Epoch 3155/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3641 - accuracy: 0.8648
    Epoch 3156/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3840 - accuracy: 0.8485
    Epoch 3157/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3588 - accuracy: 0.8648
    Epoch 3158/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3716 - accuracy: 0.8593
    Epoch 3159/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4041 - accuracy: 0.8439
    Epoch 3160/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3616 - accuracy: 0.8666
    Epoch 3161/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3683 - accuracy: 0.8612
    Epoch 3162/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3536 - accuracy: 0.8630
    Epoch 3163/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4371 - accuracy: 0.8512
    Epoch 3164/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4345 - accuracy: 0.8240
    Epoch 3165/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3661 - accuracy: 0.8612
    Epoch 3166/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3593 - accuracy: 0.8657
    Epoch 3167/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3538 - accuracy: 0.8593
    Epoch 3168/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3475 - accuracy: 0.8684
    Epoch 3169/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3555 - accuracy: 0.8730
    Epoch 3170/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3725 - accuracy: 0.8457
    Epoch 3171/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5395 - accuracy: 0.8203
    Epoch 3172/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5979 - accuracy: 0.8249
    Epoch 3173/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3630 - accuracy: 0.8575
    Epoch 3174/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3503 - accuracy: 0.8630
    Epoch 3175/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3410 - accuracy: 0.8748
    Epoch 3176/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3559 - accuracy: 0.8639
    Epoch 3177/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3780 - accuracy: 0.8630
    Epoch 3178/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3789 - accuracy: 0.8539
    Epoch 3179/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3647 - accuracy: 0.8548
    Epoch 3180/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3623 - accuracy: 0.8593
    Epoch 3181/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3583 - accuracy: 0.8593
    Epoch 3182/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3603 - accuracy: 0.8548
    Epoch 3183/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4058 - accuracy: 0.8330
    Epoch 3184/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5432 - accuracy: 0.8058
    Epoch 3185/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.5485 - accuracy: 0.8276
    Epoch 3186/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3470 - accuracy: 0.8684
    Epoch 3187/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3463 - accuracy: 0.8675
    Epoch 3188/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3441 - accuracy: 0.8666
    Epoch 3189/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3515 - accuracy: 0.8630
    Epoch 3190/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3586 - accuracy: 0.8603
    Epoch 3191/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3909 - accuracy: 0.8584
    Epoch 3192/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4677 - accuracy: 0.8276
    Epoch 3193/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3733 - accuracy: 0.8603
    Epoch 3194/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5341 - accuracy: 0.8212
    Epoch 3195/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3718 - accuracy: 0.8530
    Epoch 3196/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8721
    Epoch 3197/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3519 - accuracy: 0.8711
    Epoch 3198/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3475 - accuracy: 0.8666
    Epoch 3199/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3822 - accuracy: 0.8503
    Epoch 3200/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3800 - accuracy: 0.8612
    Epoch 3201/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3798 - accuracy: 0.8503
    Epoch 3202/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4805 - accuracy: 0.8376
    Epoch 3203/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3578 - accuracy: 0.8593
    Epoch 3204/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3454 - accuracy: 0.8684
    Epoch 3205/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4329 - accuracy: 0.8494
    Epoch 3206/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3788 - accuracy: 0.8485
    Epoch 3207/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3725 - accuracy: 0.8575
    Epoch 3208/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3715 - accuracy: 0.8603
    Epoch 3209/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3588 - accuracy: 0.8639
    Epoch 3210/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3771 - accuracy: 0.8466
    Epoch 3211/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3389 - accuracy: 0.8721
    Epoch 3212/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3397 - accuracy: 0.8684
    Epoch 3213/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4274 - accuracy: 0.8330
    Epoch 3214/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4155 - accuracy: 0.8421
    Epoch 3215/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3621 - accuracy: 0.8603
    Epoch 3216/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3957 - accuracy: 0.8566
    Epoch 3217/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3491 - accuracy: 0.8666
    Epoch 3218/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3595 - accuracy: 0.8621
    Epoch 3219/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3704 - accuracy: 0.8512
    Epoch 3220/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3561 - accuracy: 0.8648
    Epoch 3221/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4405 - accuracy: 0.8330
    Epoch 3222/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4127 - accuracy: 0.8521
    Epoch 3223/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3783 - accuracy: 0.8557
    Epoch 3224/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3907 - accuracy: 0.8539
    Epoch 3225/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3428 - accuracy: 0.8693
    Epoch 3226/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4400 - accuracy: 0.8348
    Epoch 3227/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3405 - accuracy: 0.8748
    Epoch 3228/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3678 - accuracy: 0.8702
    Epoch 3229/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3661 - accuracy: 0.8593
    Epoch 3230/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3585 - accuracy: 0.8603
    Epoch 3231/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3612 - accuracy: 0.8612
    Epoch 3232/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5012 - accuracy: 0.8321
    Epoch 3233/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5594 - accuracy: 0.8176
    Epoch 3234/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3521 - accuracy: 0.8648
    Epoch 3235/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3550 - accuracy: 0.8584
    Epoch 3236/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3558 - accuracy: 0.8630
    Epoch 3237/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4225 - accuracy: 0.8385
    Epoch 3238/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3713 - accuracy: 0.8603
    Epoch 3239/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3417 - accuracy: 0.8693
    Epoch 3240/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3877 - accuracy: 0.8566
    Epoch 3241/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3728 - accuracy: 0.8621
    Epoch 3242/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3469 - accuracy: 0.8648
    Epoch 3243/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3763 - accuracy: 0.8539
    Epoch 3244/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3691 - accuracy: 0.8575
    Epoch 3245/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3560 - accuracy: 0.8566
    Epoch 3246/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3584 - accuracy: 0.8675
    Epoch 3247/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3849 - accuracy: 0.8566
    Epoch 3248/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3572 - accuracy: 0.8621
    Epoch 3249/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3387 - accuracy: 0.8711
    Epoch 3250/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3763 - accuracy: 0.8485
    Epoch 3251/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4100 - accuracy: 0.8566
    Epoch 3252/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4272 - accuracy: 0.8412
    Epoch 3253/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3459 - accuracy: 0.8657
    Epoch 3254/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3619 - accuracy: 0.8612
    Epoch 3255/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3382 - accuracy: 0.8739
    Epoch 3256/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.5028 - accuracy: 0.8267
    Epoch 3257/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3447 - accuracy: 0.8739
    Epoch 3258/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3477 - accuracy: 0.8702
    Epoch 3259/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4326 - accuracy: 0.8412
    Epoch 3260/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3452 - accuracy: 0.8702
    Epoch 3261/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3364 - accuracy: 0.8666
    Epoch 3262/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3702 - accuracy: 0.8539
    Epoch 3263/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3746 - accuracy: 0.8593
    Epoch 3264/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3491 - accuracy: 0.8702
    Epoch 3265/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3464 - accuracy: 0.8721
    Epoch 3266/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4774 - accuracy: 0.8258
    Epoch 3267/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3964 - accuracy: 0.8530
    Epoch 3268/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4022 - accuracy: 0.8584
    Epoch 3269/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3899 - accuracy: 0.8566
    Epoch 3270/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3551 - accuracy: 0.8621
    Epoch 3271/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3795 - accuracy: 0.8494
    Epoch 3272/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3814 - accuracy: 0.8584
    Epoch 3273/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4476 - accuracy: 0.8240
    Epoch 3274/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4084 - accuracy: 0.8494
    Epoch 3275/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4210 - accuracy: 0.8439
    Epoch 3276/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4533 - accuracy: 0.8348
    Epoch 3277/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3786 - accuracy: 0.8584
    Epoch 3278/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3493 - accuracy: 0.8666
    Epoch 3279/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3451 - accuracy: 0.8675
    Epoch 3280/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3432 - accuracy: 0.8639
    Epoch 3281/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4082 - accuracy: 0.8512
    Epoch 3282/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3522 - accuracy: 0.8630
    Epoch 3283/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3518 - accuracy: 0.8702
    Epoch 3284/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3446 - accuracy: 0.8675
    Epoch 3285/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3478 - accuracy: 0.8666
    Epoch 3286/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3355 - accuracy: 0.8702
    Epoch 3287/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3613 - accuracy: 0.8739
    Epoch 3288/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3466 - accuracy: 0.8684
    Epoch 3289/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3436 - accuracy: 0.8648
    Epoch 3290/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3602 - accuracy: 0.8648
    Epoch 3291/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3626 - accuracy: 0.8603
    Epoch 3292/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3576 - accuracy: 0.8639
    Epoch 3293/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8702
    Epoch 3294/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3801 - accuracy: 0.8593
    Epoch 3295/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4434 - accuracy: 0.8457
    Epoch 3296/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3837 - accuracy: 0.8657
    Epoch 3297/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4044 - accuracy: 0.8439
    Epoch 3298/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3534 - accuracy: 0.8675
    Epoch 3299/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4012 - accuracy: 0.8367
    Epoch 3300/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3479 - accuracy: 0.8775
    Epoch 3301/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3668 - accuracy: 0.8557
    Epoch 3302/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3594 - accuracy: 0.8639
    Epoch 3303/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3438 - accuracy: 0.8666
    Epoch 3304/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3590 - accuracy: 0.8548
    Epoch 3305/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3883 - accuracy: 0.8475
    Epoch 3306/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3607 - accuracy: 0.8612
    Epoch 3307/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3916 - accuracy: 0.8521
    Epoch 3308/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3509 - accuracy: 0.8648
    Epoch 3309/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3714 - accuracy: 0.8575
    Epoch 3310/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3809 - accuracy: 0.8494
    Epoch 3311/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3583 - accuracy: 0.8675
    Epoch 3312/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3523 - accuracy: 0.8684
    Epoch 3313/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3544 - accuracy: 0.8684
    Epoch 3314/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4025 - accuracy: 0.8358
    Epoch 3315/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3641 - accuracy: 0.8666
    Epoch 3316/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3279 - accuracy: 0.8766
    Epoch 3317/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4337 - accuracy: 0.8494
    Epoch 3318/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4423 - accuracy: 0.8485
    Epoch 3319/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3630 - accuracy: 0.8557
    Epoch 3320/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3510 - accuracy: 0.8612
    Epoch 3321/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3602 - accuracy: 0.8584
    Epoch 3322/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3832 - accuracy: 0.8521
    Epoch 3323/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3678 - accuracy: 0.8530
    Epoch 3324/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3527 - accuracy: 0.8711
    Epoch 3325/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3629 - accuracy: 0.8621
    Epoch 3326/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3482 - accuracy: 0.8666
    Epoch 3327/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3470 - accuracy: 0.8666
    Epoch 3328/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3442 - accuracy: 0.8657
    Epoch 3329/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3710 - accuracy: 0.8575
    Epoch 3330/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4340 - accuracy: 0.8358
    Epoch 3331/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3471 - accuracy: 0.8666
    Epoch 3332/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3515 - accuracy: 0.8730
    Epoch 3333/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3474 - accuracy: 0.8566
    Epoch 3334/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3661 - accuracy: 0.8557
    Epoch 3335/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3480 - accuracy: 0.8684
    Epoch 3336/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3789 - accuracy: 0.8557
    Epoch 3337/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4717 - accuracy: 0.8457
    Epoch 3338/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3735 - accuracy: 0.8593
    Epoch 3339/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3582 - accuracy: 0.8584
    Epoch 3340/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3401 - accuracy: 0.8621
    Epoch 3341/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3610 - accuracy: 0.8693
    Epoch 3342/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3359 - accuracy: 0.8784
    Epoch 3343/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3885 - accuracy: 0.8584
    Epoch 3344/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3574 - accuracy: 0.8684
    Epoch 3345/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3959 - accuracy: 0.8376
    Epoch 3346/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4637 - accuracy: 0.8203
    Epoch 3347/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3747 - accuracy: 0.8557
    Epoch 3348/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3459 - accuracy: 0.8766
    Epoch 3349/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3573 - accuracy: 0.8666
    Epoch 3350/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3737 - accuracy: 0.8603
    Epoch 3351/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3350 - accuracy: 0.8775
    Epoch 3352/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3501 - accuracy: 0.8612
    Epoch 3353/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3739 - accuracy: 0.8557
    Epoch 3354/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8684
    Epoch 3355/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3917 - accuracy: 0.8503
    Epoch 3356/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3544 - accuracy: 0.8666
    Epoch 3357/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4222 - accuracy: 0.8339
    Epoch 3358/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3455 - accuracy: 0.8666
    Epoch 3359/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3461 - accuracy: 0.8748
    Epoch 3360/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3917 - accuracy: 0.8593
    Epoch 3361/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.4103 - accuracy: 0.8494
    Epoch 3362/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3813 - accuracy: 0.8566
    Epoch 3363/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3732 - accuracy: 0.8593
    Epoch 3364/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3459 - accuracy: 0.8666
    Epoch 3365/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3662 - accuracy: 0.8530
    Epoch 3366/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3892 - accuracy: 0.8512
    Epoch 3367/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3599 - accuracy: 0.8512
    Epoch 3368/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3873 - accuracy: 0.8621
    Epoch 3369/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3508 - accuracy: 0.8630
    Epoch 3370/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.5328 - accuracy: 0.8240
    Epoch 3371/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3697 - accuracy: 0.8693
    Epoch 3372/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3357 - accuracy: 0.8739
    Epoch 3373/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3548 - accuracy: 0.8621
    Epoch 3374/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3363 - accuracy: 0.8757
    Epoch 3375/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3468 - accuracy: 0.8612
    Epoch 3376/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3656 - accuracy: 0.8603
    Epoch 3377/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3557 - accuracy: 0.8639
    Epoch 3378/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3438 - accuracy: 0.8829
    Epoch 3379/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3507 - accuracy: 0.8675
    Epoch 3380/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4516 - accuracy: 0.8294
    Epoch 3381/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3603 - accuracy: 0.8630
    Epoch 3382/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3525 - accuracy: 0.8621
    Epoch 3383/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3403 - accuracy: 0.8711
    Epoch 3384/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3947 - accuracy: 0.8466
    Epoch 3385/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3331 - accuracy: 0.8730
    Epoch 3386/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3483 - accuracy: 0.8630
    Epoch 3387/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3405 - accuracy: 0.8730
    Epoch 3388/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3589 - accuracy: 0.8603
    Epoch 3389/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4015 - accuracy: 0.8385
    Epoch 3390/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3436 - accuracy: 0.8784
    Epoch 3391/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3467 - accuracy: 0.8630
    Epoch 3392/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3440 - accuracy: 0.8739
    Epoch 3393/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3704 - accuracy: 0.8575
    Epoch 3394/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3620 - accuracy: 0.8621
    Epoch 3395/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4740 - accuracy: 0.8294
    Epoch 3396/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4164 - accuracy: 0.8421
    Epoch 3397/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3674 - accuracy: 0.8593
    Epoch 3398/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3981 - accuracy: 0.8421
    Epoch 3399/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3718 - accuracy: 0.8575
    Epoch 3400/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3548 - accuracy: 0.8630
    Epoch 3401/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3883 - accuracy: 0.8485
    Epoch 3402/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3460 - accuracy: 0.8684
    Epoch 3403/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3703 - accuracy: 0.8575
    Epoch 3404/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3896 - accuracy: 0.8521
    Epoch 3405/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3396 - accuracy: 0.8739
    Epoch 3406/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3479 - accuracy: 0.8612
    Epoch 3407/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3429 - accuracy: 0.8739
    Epoch 3408/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3639 - accuracy: 0.8603
    Epoch 3409/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.4332 - accuracy: 0.8403
    Epoch 3410/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3456 - accuracy: 0.8666
    Epoch 3411/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3664 - accuracy: 0.8612
    Epoch 3412/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3529 - accuracy: 0.8784
    Epoch 3413/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3415 - accuracy: 0.8666
    Epoch 3414/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3532 - accuracy: 0.8693
    Epoch 3415/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4045 - accuracy: 0.8339
    Epoch 3416/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.4570 - accuracy: 0.8385
    Epoch 3417/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4106 - accuracy: 0.8258
    Epoch 3418/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3786 - accuracy: 0.8648
    Epoch 3419/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4245 - accuracy: 0.8348
    Epoch 3420/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3704 - accuracy: 0.8603
    Epoch 3421/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3372 - accuracy: 0.8711
    Epoch 3422/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3659 - accuracy: 0.8684
    Epoch 3423/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3731 - accuracy: 0.8603
    Epoch 3424/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3367 - accuracy: 0.8675
    Epoch 3425/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3656 - accuracy: 0.8584
    Epoch 3426/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4405 - accuracy: 0.8412
    Epoch 3427/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3582 - accuracy: 0.8566
    Epoch 3428/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3472 - accuracy: 0.8648
    Epoch 3429/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3544 - accuracy: 0.8648
    Epoch 3430/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3520 - accuracy: 0.8575
    Epoch 3431/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3646 - accuracy: 0.8612
    Epoch 3432/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3479 - accuracy: 0.8675
    Epoch 3433/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3679 - accuracy: 0.8648
    Epoch 3434/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3723 - accuracy: 0.8593
    Epoch 3435/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4341 - accuracy: 0.8412
    Epoch 3436/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4351 - accuracy: 0.8249
    Epoch 3437/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3547 - accuracy: 0.8575
    Epoch 3438/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3383 - accuracy: 0.8675
    Epoch 3439/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3598 - accuracy: 0.8566
    Epoch 3440/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3412 - accuracy: 0.8666
    Epoch 3441/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4434 - accuracy: 0.8439
    Epoch 3442/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3612 - accuracy: 0.8612
    Epoch 3443/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3631 - accuracy: 0.8575
    Epoch 3444/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3471 - accuracy: 0.8711
    Epoch 3445/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3442 - accuracy: 0.8630
    Epoch 3446/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3527 - accuracy: 0.8612
    Epoch 3447/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3336 - accuracy: 0.8775
    Epoch 3448/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3454 - accuracy: 0.8702
    Epoch 3449/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3613 - accuracy: 0.8630
    Epoch 3450/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3415 - accuracy: 0.8711
    Epoch 3451/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3322 - accuracy: 0.8793
    Epoch 3452/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3396 - accuracy: 0.8693
    Epoch 3453/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3371 - accuracy: 0.8702
    Epoch 3454/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3485 - accuracy: 0.8657
    Epoch 3455/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3525 - accuracy: 0.8657
    Epoch 3456/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3331 - accuracy: 0.8721
    Epoch 3457/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3573 - accuracy: 0.8603
    Epoch 3458/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3595 - accuracy: 0.8593
    Epoch 3459/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3750 - accuracy: 0.8648
    Epoch 3460/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3444 - accuracy: 0.8675
    Epoch 3461/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3467 - accuracy: 0.8666
    Epoch 3462/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3440 - accuracy: 0.8711
    Epoch 3463/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3604 - accuracy: 0.8603
    Epoch 3464/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4204 - accuracy: 0.8358
    Epoch 3465/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3612 - accuracy: 0.8630
    Epoch 3466/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3448 - accuracy: 0.8648
    Epoch 3467/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3787 - accuracy: 0.8539
    Epoch 3468/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4480 - accuracy: 0.8303
    Epoch 3469/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4031 - accuracy: 0.8439
    Epoch 3470/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3462 - accuracy: 0.8829
    Epoch 3471/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3601 - accuracy: 0.8603
    Epoch 3472/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3664 - accuracy: 0.8612
    Epoch 3473/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3436 - accuracy: 0.8693
    Epoch 3474/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3590 - accuracy: 0.8666
    Epoch 3475/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3453 - accuracy: 0.8721
    Epoch 3476/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3522 - accuracy: 0.8702
    Epoch 3477/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4118 - accuracy: 0.8385
    Epoch 3478/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3826 - accuracy: 0.8503
    Epoch 3479/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3361 - accuracy: 0.8693
    Epoch 3480/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3567 - accuracy: 0.8557
    Epoch 3481/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3466 - accuracy: 0.8612
    Epoch 3482/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3568 - accuracy: 0.8612
    Epoch 3483/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3406 - accuracy: 0.8702
    Epoch 3484/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3831 - accuracy: 0.8584
    Epoch 3485/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3592 - accuracy: 0.8584
    Epoch 3486/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3546 - accuracy: 0.8648
    Epoch 3487/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3447 - accuracy: 0.8702
    Epoch 3488/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3903 - accuracy: 0.8512
    Epoch 3489/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3538 - accuracy: 0.8675
    Epoch 3490/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3969 - accuracy: 0.8485
    Epoch 3491/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3633 - accuracy: 0.8621
    Epoch 3492/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3334 - accuracy: 0.8675
    Epoch 3493/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3418 - accuracy: 0.8693
    Epoch 3494/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3470 - accuracy: 0.8666
    Epoch 3495/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3669 - accuracy: 0.8557
    Epoch 3496/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3656 - accuracy: 0.8530
    Epoch 3497/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3400 - accuracy: 0.8657
    Epoch 3498/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4432 - accuracy: 0.8230
    Epoch 3499/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3840 - accuracy: 0.8512
    Epoch 3500/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3388 - accuracy: 0.8639
    Epoch 3501/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3496 - accuracy: 0.8721
    Epoch 3502/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3590 - accuracy: 0.8584
    Epoch 3503/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3465 - accuracy: 0.8648
    Epoch 3504/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3420 - accuracy: 0.8630
    Epoch 3505/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3711 - accuracy: 0.8621
    Epoch 3506/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3568 - accuracy: 0.8603
    Epoch 3507/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3406 - accuracy: 0.8666
    Epoch 3508/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3509 - accuracy: 0.8639
    Epoch 3509/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3437 - accuracy: 0.8702
    Epoch 3510/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3843 - accuracy: 0.8475
    Epoch 3511/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3380 - accuracy: 0.8657
    Epoch 3512/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4201 - accuracy: 0.8403
    Epoch 3513/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3637 - accuracy: 0.8566
    Epoch 3514/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3534 - accuracy: 0.8730
    Epoch 3515/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3337 - accuracy: 0.8639
    Epoch 3516/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3732 - accuracy: 0.8530
    Epoch 3517/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3289 - accuracy: 0.8730
    Epoch 3518/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3859 - accuracy: 0.8503
    Epoch 3519/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3357 - accuracy: 0.8693
    Epoch 3520/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3514 - accuracy: 0.8657
    Epoch 3521/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3459 - accuracy: 0.8603
    Epoch 3522/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3367 - accuracy: 0.8666
    Epoch 3523/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3633 - accuracy: 0.8648
    Epoch 3524/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3563 - accuracy: 0.8675
    Epoch 3525/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3434 - accuracy: 0.8684
    Epoch 3526/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3859 - accuracy: 0.8512
    Epoch 3527/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3652 - accuracy: 0.8575
    Epoch 3528/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3971 - accuracy: 0.8539
    Epoch 3529/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3919 - accuracy: 0.8412
    Epoch 3530/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4483 - accuracy: 0.8348
    Epoch 3531/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3852 - accuracy: 0.8575
    Epoch 3532/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3659 - accuracy: 0.8575
    Epoch 3533/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3542 - accuracy: 0.8657
    Epoch 3534/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3405 - accuracy: 0.8693
    Epoch 3535/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3571 - accuracy: 0.8630
    Epoch 3536/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3674 - accuracy: 0.8566
    Epoch 3537/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3363 - accuracy: 0.8684
    Epoch 3538/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3432 - accuracy: 0.8621
    Epoch 3539/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3340 - accuracy: 0.8693
    Epoch 3540/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3702 - accuracy: 0.8539
    Epoch 3541/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3669 - accuracy: 0.8575
    Epoch 3542/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3759 - accuracy: 0.8575
    Epoch 3543/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3384 - accuracy: 0.8693
    Epoch 3544/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3591 - accuracy: 0.8648
    Epoch 3545/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4130 - accuracy: 0.8339
    Epoch 3546/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3837 - accuracy: 0.8448
    Epoch 3547/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3693 - accuracy: 0.8575
    Epoch 3548/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3655 - accuracy: 0.8512
    Epoch 3549/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3395 - accuracy: 0.8693
    Epoch 3550/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3482 - accuracy: 0.8657
    Epoch 3551/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3820 - accuracy: 0.8548
    Epoch 3552/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3475 - accuracy: 0.8621
    Epoch 3553/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3812 - accuracy: 0.8548
    Epoch 3554/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3455 - accuracy: 0.8648
    Epoch 3555/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3517 - accuracy: 0.8603
    Epoch 3556/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3408 - accuracy: 0.8711
    Epoch 3557/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3581 - accuracy: 0.8612
    Epoch 3558/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3533 - accuracy: 0.8584
    Epoch 3559/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3621 - accuracy: 0.8593
    Epoch 3560/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3374 - accuracy: 0.8757
    Epoch 3561/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3398 - accuracy: 0.8639
    Epoch 3562/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3464 - accuracy: 0.8648
    Epoch 3563/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3609 - accuracy: 0.8675
    Epoch 3564/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3935 - accuracy: 0.8503
    Epoch 3565/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3717 - accuracy: 0.8603
    Epoch 3566/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3567 - accuracy: 0.8648
    Epoch 3567/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3390 - accuracy: 0.8684
    Epoch 3568/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3539 - accuracy: 0.8575
    Epoch 3569/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3752 - accuracy: 0.8475
    Epoch 3570/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3379 - accuracy: 0.8675
    Epoch 3571/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3550 - accuracy: 0.8666
    Epoch 3572/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3981 - accuracy: 0.8376
    Epoch 3573/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3630 - accuracy: 0.8666
    Epoch 3574/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3815 - accuracy: 0.8557
    Epoch 3575/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8666
    Epoch 3576/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3561 - accuracy: 0.8603
    Epoch 3577/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3360 - accuracy: 0.8711
    Epoch 3578/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3634 - accuracy: 0.8593
    Epoch 3579/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3733 - accuracy: 0.8521
    Epoch 3580/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3910 - accuracy: 0.8530
    Epoch 3581/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3817 - accuracy: 0.8666
    Epoch 3582/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4158 - accuracy: 0.8448
    Epoch 3583/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3579 - accuracy: 0.8548
    Epoch 3584/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4021 - accuracy: 0.8421
    Epoch 3585/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3662 - accuracy: 0.8566
    Epoch 3586/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3549 - accuracy: 0.8603
    Epoch 3587/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3509 - accuracy: 0.8657
    Epoch 3588/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3496 - accuracy: 0.8648
    Epoch 3589/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3718 - accuracy: 0.8621
    Epoch 3590/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3627 - accuracy: 0.8621
    Epoch 3591/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3380 - accuracy: 0.8684
    Epoch 3592/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3645 - accuracy: 0.8566
    Epoch 3593/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3934 - accuracy: 0.8475
    Epoch 3594/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3592 - accuracy: 0.8612
    Epoch 3595/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3444 - accuracy: 0.8675
    Epoch 3596/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3521 - accuracy: 0.8666
    Epoch 3597/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3440 - accuracy: 0.8612
    Epoch 3598/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4021 - accuracy: 0.8512
    Epoch 3599/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3463 - accuracy: 0.8630
    Epoch 3600/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3575 - accuracy: 0.8539
    Epoch 3601/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3880 - accuracy: 0.8557
    Epoch 3602/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3584 - accuracy: 0.8557
    Epoch 3603/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3432 - accuracy: 0.8675
    Epoch 3604/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3668 - accuracy: 0.8575
    Epoch 3605/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3896 - accuracy: 0.8548
    Epoch 3606/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3528 - accuracy: 0.8566
    Epoch 3607/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3932 - accuracy: 0.8430
    Epoch 3608/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3584 - accuracy: 0.8557
    Epoch 3609/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3896 - accuracy: 0.8539
    Epoch 3610/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3689 - accuracy: 0.8539
    Epoch 3611/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3435 - accuracy: 0.8684
    Epoch 3612/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3608 - accuracy: 0.8675
    Epoch 3613/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3417 - accuracy: 0.8657
    Epoch 3614/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3670 - accuracy: 0.8584
    Epoch 3615/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3424 - accuracy: 0.8693
    Epoch 3616/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3692 - accuracy: 0.8539
    Epoch 3617/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3900 - accuracy: 0.8539
    Epoch 3618/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3638 - accuracy: 0.8557
    Epoch 3619/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3530 - accuracy: 0.8684
    Epoch 3620/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3484 - accuracy: 0.8648
    Epoch 3621/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3582 - accuracy: 0.8639
    Epoch 3622/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3460 - accuracy: 0.8648
    Epoch 3623/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3480 - accuracy: 0.8593
    Epoch 3624/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3599 - accuracy: 0.8639
    Epoch 3625/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3478 - accuracy: 0.8693
    Epoch 3626/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3565 - accuracy: 0.8675
    Epoch 3627/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3540 - accuracy: 0.8593
    Epoch 3628/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3510 - accuracy: 0.8630
    Epoch 3629/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3672 - accuracy: 0.8584
    Epoch 3630/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3538 - accuracy: 0.8566
    Epoch 3631/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3934 - accuracy: 0.8512
    Epoch 3632/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3517 - accuracy: 0.8575
    Epoch 3633/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3532 - accuracy: 0.8693
    Epoch 3634/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3379 - accuracy: 0.8684
    Epoch 3635/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3479 - accuracy: 0.8739
    Epoch 3636/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3451 - accuracy: 0.8593
    Epoch 3637/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3731 - accuracy: 0.8566
    Epoch 3638/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3602 - accuracy: 0.8575
    Epoch 3639/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3610 - accuracy: 0.8557
    Epoch 3640/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3417 - accuracy: 0.8657
    Epoch 3641/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3649 - accuracy: 0.8557
    Epoch 3642/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3515 - accuracy: 0.8657
    Epoch 3643/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3840 - accuracy: 0.8530
    Epoch 3644/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3447 - accuracy: 0.8721
    Epoch 3645/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4008 - accuracy: 0.8530
    Epoch 3646/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3441 - accuracy: 0.8684
    Epoch 3647/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3331 - accuracy: 0.8684
    Epoch 3648/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3344 - accuracy: 0.8739
    Epoch 3649/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3890 - accuracy: 0.8530
    Epoch 3650/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3513 - accuracy: 0.8657
    Epoch 3651/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3672 - accuracy: 0.8593
    Epoch 3652/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3389 - accuracy: 0.8675
    Epoch 3653/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3511 - accuracy: 0.8657
    Epoch 3654/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3395 - accuracy: 0.8630
    Epoch 3655/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3470 - accuracy: 0.8603
    Epoch 3656/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3630 - accuracy: 0.8503
    Epoch 3657/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3603 - accuracy: 0.8584
    Epoch 3658/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3993 - accuracy: 0.8394
    Epoch 3659/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3518 - accuracy: 0.8648
    Epoch 3660/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3509 - accuracy: 0.8612
    Epoch 3661/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3396 - accuracy: 0.8730
    Epoch 3662/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3662 - accuracy: 0.8584
    Epoch 3663/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3546 - accuracy: 0.8657
    Epoch 3664/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3373 - accuracy: 0.8748
    Epoch 3665/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3727 - accuracy: 0.8566
    Epoch 3666/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3713 - accuracy: 0.8557
    Epoch 3667/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3602 - accuracy: 0.8593
    Epoch 3668/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3467 - accuracy: 0.8693
    Epoch 3669/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3556 - accuracy: 0.8603
    Epoch 3670/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3553 - accuracy: 0.8603
    Epoch 3671/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3473 - accuracy: 0.8675
    Epoch 3672/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3784 - accuracy: 0.8584
    Epoch 3673/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3397 - accuracy: 0.8639
    Epoch 3674/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3444 - accuracy: 0.8657
    Epoch 3675/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3552 - accuracy: 0.8639
    Epoch 3676/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3509 - accuracy: 0.8603
    Epoch 3677/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3450 - accuracy: 0.8612
    Epoch 3678/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3483 - accuracy: 0.8603
    Epoch 3679/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3435 - accuracy: 0.8639
    Epoch 3680/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3400 - accuracy: 0.8748
    Epoch 3681/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3645 - accuracy: 0.8566
    Epoch 3682/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3404 - accuracy: 0.8684
    Epoch 3683/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3437 - accuracy: 0.8693
    Epoch 3684/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3902 - accuracy: 0.8603
    Epoch 3685/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3755 - accuracy: 0.8448
    Epoch 3686/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3849 - accuracy: 0.8521
    Epoch 3687/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3376 - accuracy: 0.8666
    Epoch 3688/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3498 - accuracy: 0.8666
    Epoch 3689/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3726 - accuracy: 0.8630
    Epoch 3690/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3464 - accuracy: 0.8675
    Epoch 3691/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3422 - accuracy: 0.8702
    Epoch 3692/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3328 - accuracy: 0.8648
    Epoch 3693/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3415 - accuracy: 0.8711
    Epoch 3694/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3481 - accuracy: 0.8603
    Epoch 3695/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3570 - accuracy: 0.8539
    Epoch 3696/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3846 - accuracy: 0.8494
    Epoch 3697/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3436 - accuracy: 0.8684
    Epoch 3698/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3627 - accuracy: 0.8557
    Epoch 3699/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3636 - accuracy: 0.8621
    Epoch 3700/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3446 - accuracy: 0.8711
    Epoch 3701/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3577 - accuracy: 0.8584
    Epoch 3702/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3406 - accuracy: 0.8684
    Epoch 3703/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3438 - accuracy: 0.8657
    Epoch 3704/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3523 - accuracy: 0.8657
    Epoch 3705/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3637 - accuracy: 0.8648
    Epoch 3706/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3721 - accuracy: 0.8702
    Epoch 3707/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3829 - accuracy: 0.8603
    Epoch 3708/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3759 - accuracy: 0.8593
    Epoch 3709/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3983 - accuracy: 0.8421
    Epoch 3710/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3785 - accuracy: 0.8530
    Epoch 3711/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3505 - accuracy: 0.8584
    Epoch 3712/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3393 - accuracy: 0.8711
    Epoch 3713/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3648 - accuracy: 0.8621
    Epoch 3714/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3860 - accuracy: 0.8503
    Epoch 3715/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3691 - accuracy: 0.8593
    Epoch 3716/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3581 - accuracy: 0.8603
    Epoch 3717/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3605 - accuracy: 0.8566
    Epoch 3718/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3405 - accuracy: 0.8666
    Epoch 3719/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3725 - accuracy: 0.8557
    Epoch 3720/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3486 - accuracy: 0.8657
    Epoch 3721/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3501 - accuracy: 0.8657
    Epoch 3722/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3380 - accuracy: 0.8702
    Epoch 3723/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3498 - accuracy: 0.8593
    Epoch 3724/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3931 - accuracy: 0.8421
    Epoch 3725/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3485 - accuracy: 0.8666
    Epoch 3726/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3483 - accuracy: 0.8648
    Epoch 3727/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3466 - accuracy: 0.8675
    Epoch 3728/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3369 - accuracy: 0.8666
    Epoch 3729/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3893 - accuracy: 0.8548
    Epoch 3730/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3562 - accuracy: 0.8648
    Epoch 3731/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3572 - accuracy: 0.8630
    Epoch 3732/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3490 - accuracy: 0.8530
    Epoch 3733/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3657 - accuracy: 0.8494
    Epoch 3734/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3585 - accuracy: 0.8621
    Epoch 3735/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3414 - accuracy: 0.8684
    Epoch 3736/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3421 - accuracy: 0.8684
    Epoch 3737/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3456 - accuracy: 0.8657
    Epoch 3738/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3421 - accuracy: 0.8639
    Epoch 3739/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3462 - accuracy: 0.8721
    Epoch 3740/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3395 - accuracy: 0.8702
    Epoch 3741/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3377 - accuracy: 0.8657
    Epoch 3742/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3437 - accuracy: 0.8748
    Epoch 3743/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3717 - accuracy: 0.8530
    Epoch 3744/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3886 - accuracy: 0.8603
    Epoch 3745/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3638 - accuracy: 0.8584
    Epoch 3746/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3638 - accuracy: 0.8639
    Epoch 3747/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3456 - accuracy: 0.8666
    Epoch 3748/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3506 - accuracy: 0.8575
    Epoch 3749/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3503 - accuracy: 0.8603
    Epoch 3750/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3351 - accuracy: 0.8675
    Epoch 3751/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3648 - accuracy: 0.8639
    Epoch 3752/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3696 - accuracy: 0.8584
    Epoch 3753/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3433 - accuracy: 0.8593
    Epoch 3754/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3482 - accuracy: 0.8603
    Epoch 3755/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3750 - accuracy: 0.8512
    Epoch 3756/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3534 - accuracy: 0.8657
    Epoch 3757/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3423 - accuracy: 0.8593
    Epoch 3758/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3594 - accuracy: 0.8621
    Epoch 3759/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3854 - accuracy: 0.8530
    Epoch 3760/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3720 - accuracy: 0.8530
    Epoch 3761/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3423 - accuracy: 0.8648
    Epoch 3762/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3407 - accuracy: 0.8648
    Epoch 3763/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3820 - accuracy: 0.8557
    Epoch 3764/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3574 - accuracy: 0.8584
    Epoch 3765/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3586 - accuracy: 0.8593
    Epoch 3766/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3384 - accuracy: 0.8693
    Epoch 3767/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3480 - accuracy: 0.8603
    Epoch 3768/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3492 - accuracy: 0.8648
    Epoch 3769/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3546 - accuracy: 0.8603
    Epoch 3770/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3377 - accuracy: 0.8684
    Epoch 3771/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3768 - accuracy: 0.8584
    Epoch 3772/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3477 - accuracy: 0.8630
    Epoch 3773/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3371 - accuracy: 0.8711
    Epoch 3774/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3539 - accuracy: 0.8666
    Epoch 3775/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3361 - accuracy: 0.8711
    Epoch 3776/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3374 - accuracy: 0.8748
    Epoch 3777/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3440 - accuracy: 0.8639
    Epoch 3778/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3481 - accuracy: 0.8639
    Epoch 3779/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3773 - accuracy: 0.8530
    Epoch 3780/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3533 - accuracy: 0.8621
    Epoch 3781/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3402 - accuracy: 0.8648
    Epoch 3782/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3587 - accuracy: 0.8566
    Epoch 3783/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3433 - accuracy: 0.8693
    Epoch 3784/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3712 - accuracy: 0.8503
    Epoch 3785/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3491 - accuracy: 0.8639
    Epoch 3786/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3451 - accuracy: 0.8657
    Epoch 3787/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3577 - accuracy: 0.8603
    Epoch 3788/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3549 - accuracy: 0.8584
    Epoch 3789/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3571 - accuracy: 0.8612
    Epoch 3790/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3483 - accuracy: 0.8711
    Epoch 3791/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3509 - accuracy: 0.8575
    Epoch 3792/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3505 - accuracy: 0.8593
    Epoch 3793/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3553 - accuracy: 0.8675
    Epoch 3794/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3320 - accuracy: 0.8666
    Epoch 3795/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3542 - accuracy: 0.8621
    Epoch 3796/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3856 - accuracy: 0.8530
    Epoch 3797/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3452 - accuracy: 0.8657
    Epoch 3798/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3479 - accuracy: 0.8684
    Epoch 3799/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3659 - accuracy: 0.8584
    Epoch 3800/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3510 - accuracy: 0.8584
    Epoch 3801/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3362 - accuracy: 0.8666
    Epoch 3802/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3403 - accuracy: 0.8693
    Epoch 3803/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3509 - accuracy: 0.8666
    Epoch 3804/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3426 - accuracy: 0.8675
    Epoch 3805/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3398 - accuracy: 0.8721
    Epoch 3806/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3423 - accuracy: 0.8684
    Epoch 3807/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3639 - accuracy: 0.8521
    Epoch 3808/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3525 - accuracy: 0.8657
    Epoch 3809/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3580 - accuracy: 0.8566
    Epoch 3810/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3574 - accuracy: 0.8557
    Epoch 3811/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3474 - accuracy: 0.8730
    Epoch 3812/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3433 - accuracy: 0.8702
    Epoch 3813/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3489 - accuracy: 0.8693
    Epoch 3814/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3356 - accuracy: 0.8684
    Epoch 3815/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3505 - accuracy: 0.8621
    Epoch 3816/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3586 - accuracy: 0.8630
    Epoch 3817/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3637 - accuracy: 0.8566
    Epoch 3818/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3903 - accuracy: 0.8457
    Epoch 3819/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3447 - accuracy: 0.8702
    Epoch 3820/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3546 - accuracy: 0.8557
    Epoch 3821/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3577 - accuracy: 0.8693
    Epoch 3822/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3402 - accuracy: 0.8675
    Epoch 3823/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3486 - accuracy: 0.8675
    Epoch 3824/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3572 - accuracy: 0.8621
    Epoch 3825/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3462 - accuracy: 0.8675
    Epoch 3826/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3895 - accuracy: 0.8584
    Epoch 3827/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3397 - accuracy: 0.8702
    Epoch 3828/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3453 - accuracy: 0.8621
    Epoch 3829/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3448 - accuracy: 0.8702
    Epoch 3830/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3710 - accuracy: 0.8593
    Epoch 3831/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3422 - accuracy: 0.8675
    Epoch 3832/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3480 - accuracy: 0.8639
    Epoch 3833/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3448 - accuracy: 0.8675
    Epoch 3834/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3527 - accuracy: 0.8666
    Epoch 3835/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3451 - accuracy: 0.8603
    Epoch 3836/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3468 - accuracy: 0.8730
    Epoch 3837/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3658 - accuracy: 0.8603
    Epoch 3838/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3496 - accuracy: 0.8684
    Epoch 3839/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3465 - accuracy: 0.8711
    Epoch 3840/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3433 - accuracy: 0.8593
    Epoch 3841/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3431 - accuracy: 0.8666
    Epoch 3842/6000
    35/35 [==============================] - 0s 706us/step - loss: 0.3583 - accuracy: 0.8675
    Epoch 3843/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3538 - accuracy: 0.8621
    Epoch 3844/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3566 - accuracy: 0.8639
    Epoch 3845/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3384 - accuracy: 0.8739
    Epoch 3846/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3499 - accuracy: 0.8748
    Epoch 3847/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3397 - accuracy: 0.8702
    Epoch 3848/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3675 - accuracy: 0.8621
    Epoch 3849/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3492 - accuracy: 0.8693
    Epoch 3850/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3351 - accuracy: 0.8739
    Epoch 3851/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3328 - accuracy: 0.8657
    Epoch 3852/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3493 - accuracy: 0.8666
    Epoch 3853/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3480 - accuracy: 0.8648
    Epoch 3854/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3701 - accuracy: 0.8612
    Epoch 3855/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3384 - accuracy: 0.8702
    Epoch 3856/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3615 - accuracy: 0.8584
    Epoch 3857/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3325 - accuracy: 0.8657
    Epoch 3858/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3453 - accuracy: 0.8666
    Epoch 3859/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3755 - accuracy: 0.8575
    Epoch 3860/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3498 - accuracy: 0.8584
    Epoch 3861/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3502 - accuracy: 0.8612
    Epoch 3862/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3556 - accuracy: 0.8684
    Epoch 3863/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3551 - accuracy: 0.8621
    Epoch 3864/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3516 - accuracy: 0.8621
    Epoch 3865/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3365 - accuracy: 0.8648
    Epoch 3866/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3503 - accuracy: 0.8639
    Epoch 3867/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3444 - accuracy: 0.8684
    Epoch 3868/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3405 - accuracy: 0.8684
    Epoch 3869/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3635 - accuracy: 0.8584
    Epoch 3870/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3594 - accuracy: 0.8557
    Epoch 3871/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3388 - accuracy: 0.8693
    Epoch 3872/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3462 - accuracy: 0.8721
    Epoch 3873/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3482 - accuracy: 0.8612
    Epoch 3874/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3478 - accuracy: 0.8675
    Epoch 3875/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3441 - accuracy: 0.8684
    Epoch 3876/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4072 - accuracy: 0.8367
    Epoch 3877/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3434 - accuracy: 0.8666
    Epoch 3878/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3428 - accuracy: 0.8702
    Epoch 3879/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3411 - accuracy: 0.8721
    Epoch 3880/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3555 - accuracy: 0.8548
    Epoch 3881/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3434 - accuracy: 0.8721
    Epoch 3882/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3728 - accuracy: 0.8621
    Epoch 3883/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3439 - accuracy: 0.8621
    Epoch 3884/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3627 - accuracy: 0.8639
    Epoch 3885/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3719 - accuracy: 0.8539
    Epoch 3886/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3492 - accuracy: 0.8702
    Epoch 3887/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3414 - accuracy: 0.8675
    Epoch 3888/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3418 - accuracy: 0.8666
    Epoch 3889/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3407 - accuracy: 0.8730
    Epoch 3890/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4467 - accuracy: 0.8303
    Epoch 3891/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3858 - accuracy: 0.8584
    Epoch 3892/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3492 - accuracy: 0.8666
    Epoch 3893/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3598 - accuracy: 0.8566
    Epoch 3894/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3451 - accuracy: 0.8593
    Epoch 3895/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3411 - accuracy: 0.8603
    Epoch 3896/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3420 - accuracy: 0.8657
    Epoch 3897/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3372 - accuracy: 0.8721
    Epoch 3898/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3558 - accuracy: 0.8612
    Epoch 3899/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3801 - accuracy: 0.8512
    Epoch 3900/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3537 - accuracy: 0.8621
    Epoch 3901/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3506 - accuracy: 0.8593
    Epoch 3902/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3377 - accuracy: 0.8693
    Epoch 3903/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3389 - accuracy: 0.8730
    Epoch 3904/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3391 - accuracy: 0.8666
    Epoch 3905/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3389 - accuracy: 0.8693
    Epoch 3906/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3365 - accuracy: 0.8757
    Epoch 3907/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3706 - accuracy: 0.8548
    Epoch 3908/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3430 - accuracy: 0.8675
    Epoch 3909/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3488 - accuracy: 0.8639
    Epoch 3910/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3503 - accuracy: 0.8639
    Epoch 3911/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3532 - accuracy: 0.8666
    Epoch 3912/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3614 - accuracy: 0.8693
    Epoch 3913/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3431 - accuracy: 0.8711
    Epoch 3914/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3455 - accuracy: 0.8657
    Epoch 3915/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3507 - accuracy: 0.8711
    Epoch 3916/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.4462 - accuracy: 0.8303
    Epoch 3917/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4076 - accuracy: 0.8439
    Epoch 3918/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3423 - accuracy: 0.8739
    Epoch 3919/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3430 - accuracy: 0.8702
    Epoch 3920/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3564 - accuracy: 0.8612
    Epoch 3921/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3368 - accuracy: 0.8730
    Epoch 3922/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3351 - accuracy: 0.8721
    Epoch 3923/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3892 - accuracy: 0.8503
    Epoch 3924/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3455 - accuracy: 0.8630
    Epoch 3925/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3724 - accuracy: 0.8593
    Epoch 3926/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3547 - accuracy: 0.8584
    Epoch 3927/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3724 - accuracy: 0.8593
    Epoch 3928/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3798 - accuracy: 0.8566
    Epoch 3929/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3577 - accuracy: 0.8539
    Epoch 3930/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3480 - accuracy: 0.8612
    Epoch 3931/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3523 - accuracy: 0.8684
    Epoch 3932/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3389 - accuracy: 0.8693
    Epoch 3933/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3616 - accuracy: 0.8566
    Epoch 3934/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3531 - accuracy: 0.8684
    Epoch 3935/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3381 - accuracy: 0.8702
    Epoch 3936/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3864 - accuracy: 0.8457
    Epoch 3937/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3324 - accuracy: 0.8684
    Epoch 3938/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3412 - accuracy: 0.8575
    Epoch 3939/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3456 - accuracy: 0.8666
    Epoch 3940/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3495 - accuracy: 0.8621
    Epoch 3941/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3351 - accuracy: 0.8684
    Epoch 3942/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3609 - accuracy: 0.8575
    Epoch 3943/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3683 - accuracy: 0.8593
    Epoch 3944/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8711
    Epoch 3945/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3481 - accuracy: 0.8621
    Epoch 3946/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3463 - accuracy: 0.8675
    Epoch 3947/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3466 - accuracy: 0.8630
    Epoch 3948/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3463 - accuracy: 0.8648
    Epoch 3949/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3413 - accuracy: 0.8657
    Epoch 3950/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3442 - accuracy: 0.8693
    Epoch 3951/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3485 - accuracy: 0.8639
    Epoch 3952/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3505 - accuracy: 0.8666
    Epoch 3953/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3652 - accuracy: 0.8584
    Epoch 3954/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3538 - accuracy: 0.8593
    Epoch 3955/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3736 - accuracy: 0.8657
    Epoch 3956/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3510 - accuracy: 0.8584
    Epoch 3957/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3486 - accuracy: 0.8739
    Epoch 3958/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3784 - accuracy: 0.8575
    Epoch 3959/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3380 - accuracy: 0.8612
    Epoch 3960/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3564 - accuracy: 0.8621
    Epoch 3961/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3359 - accuracy: 0.8639
    Epoch 3962/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3601 - accuracy: 0.8548
    Epoch 3963/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3443 - accuracy: 0.8584
    Epoch 3964/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3357 - accuracy: 0.8730
    Epoch 3965/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3535 - accuracy: 0.8639
    Epoch 3966/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3555 - accuracy: 0.8593
    Epoch 3967/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3398 - accuracy: 0.8684
    Epoch 3968/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3462 - accuracy: 0.8630
    Epoch 3969/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3358 - accuracy: 0.8684
    Epoch 3970/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3524 - accuracy: 0.8621
    Epoch 3971/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3819 - accuracy: 0.8530
    Epoch 3972/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3426 - accuracy: 0.8621
    Epoch 3973/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3488 - accuracy: 0.8630
    Epoch 3974/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3508 - accuracy: 0.8603
    Epoch 3975/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3512 - accuracy: 0.8593
    Epoch 3976/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3454 - accuracy: 0.8684
    Epoch 3977/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3468 - accuracy: 0.8621
    Epoch 3978/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3369 - accuracy: 0.8666
    Epoch 3979/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3931 - accuracy: 0.8448
    Epoch 3980/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3556 - accuracy: 0.8639
    Epoch 3981/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3540 - accuracy: 0.8648
    Epoch 3982/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3352 - accuracy: 0.8702
    Epoch 3983/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3340 - accuracy: 0.8721
    Epoch 3984/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3318 - accuracy: 0.8748
    Epoch 3985/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3502 - accuracy: 0.8666
    Epoch 3986/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3648 - accuracy: 0.8721
    Epoch 3987/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3537 - accuracy: 0.8603
    Epoch 3988/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3430 - accuracy: 0.8675
    Epoch 3989/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3420 - accuracy: 0.8739
    Epoch 3990/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3487 - accuracy: 0.8693
    Epoch 3991/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3928 - accuracy: 0.8421
    Epoch 3992/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3788 - accuracy: 0.8530
    Epoch 3993/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3465 - accuracy: 0.8621
    Epoch 3994/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3424 - accuracy: 0.8675
    Epoch 3995/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3416 - accuracy: 0.8621
    Epoch 3996/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3425 - accuracy: 0.8711
    Epoch 3997/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3432 - accuracy: 0.8648
    Epoch 3998/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3670 - accuracy: 0.8603
    Epoch 3999/6000
    35/35 [==============================] - 0s 706us/step - loss: 0.3524 - accuracy: 0.8684
    Epoch 4000/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3431 - accuracy: 0.8693
    Epoch 4001/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3476 - accuracy: 0.8621
    Epoch 4002/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3385 - accuracy: 0.8639
    Epoch 4003/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3395 - accuracy: 0.8584
    Epoch 4004/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3453 - accuracy: 0.8630
    Epoch 4005/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3456 - accuracy: 0.8702
    Epoch 4006/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3419 - accuracy: 0.8693
    Epoch 4007/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3471 - accuracy: 0.8639
    Epoch 4008/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3448 - accuracy: 0.8657
    Epoch 4009/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3561 - accuracy: 0.8603
    Epoch 4010/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3364 - accuracy: 0.8757
    Epoch 4011/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3694 - accuracy: 0.8666
    Epoch 4012/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3449 - accuracy: 0.8666
    Epoch 4013/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3364 - accuracy: 0.8684
    Epoch 4014/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3425 - accuracy: 0.8639
    Epoch 4015/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3498 - accuracy: 0.8639
    Epoch 4016/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3431 - accuracy: 0.8639
    Epoch 4017/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3516 - accuracy: 0.8648
    Epoch 4018/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3535 - accuracy: 0.8621
    Epoch 4019/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3811 - accuracy: 0.8584
    Epoch 4020/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3496 - accuracy: 0.8702
    Epoch 4021/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3780 - accuracy: 0.8430
    Epoch 4022/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3555 - accuracy: 0.8612
    Epoch 4023/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3583 - accuracy: 0.8648
    Epoch 4024/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3458 - accuracy: 0.8657
    Epoch 4025/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3340 - accuracy: 0.8684
    Epoch 4026/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3633 - accuracy: 0.8521
    Epoch 4027/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3411 - accuracy: 0.8657
    Epoch 4028/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3850 - accuracy: 0.8521
    Epoch 4029/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3463 - accuracy: 0.8702
    Epoch 4030/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3589 - accuracy: 0.8566
    Epoch 4031/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.4050 - accuracy: 0.8430
    Epoch 4032/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3399 - accuracy: 0.8702
    Epoch 4033/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3515 - accuracy: 0.8648
    Epoch 4034/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3801 - accuracy: 0.8475
    Epoch 4035/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3788 - accuracy: 0.8621
    Epoch 4036/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3417 - accuracy: 0.8657
    Epoch 4037/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3401 - accuracy: 0.8711
    Epoch 4038/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3466 - accuracy: 0.8730
    Epoch 4039/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3410 - accuracy: 0.8693
    Epoch 4040/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3367 - accuracy: 0.8730
    Epoch 4041/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3423 - accuracy: 0.8639
    Epoch 4042/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3741 - accuracy: 0.8566
    Epoch 4043/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3779 - accuracy: 0.8485
    Epoch 4044/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3437 - accuracy: 0.8666
    Epoch 4045/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3395 - accuracy: 0.8657
    Epoch 4046/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3410 - accuracy: 0.8711
    Epoch 4047/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3488 - accuracy: 0.8603
    Epoch 4048/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3454 - accuracy: 0.8666
    Epoch 4049/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3373 - accuracy: 0.8693
    Epoch 4050/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3401 - accuracy: 0.8603
    Epoch 4051/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3372 - accuracy: 0.8711
    Epoch 4052/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3462 - accuracy: 0.8648
    Epoch 4053/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3617 - accuracy: 0.8593
    Epoch 4054/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3674 - accuracy: 0.8584
    Epoch 4055/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3355 - accuracy: 0.8684
    Epoch 4056/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3543 - accuracy: 0.8593
    Epoch 4057/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3510 - accuracy: 0.8621
    Epoch 4058/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3422 - accuracy: 0.8684
    Epoch 4059/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3355 - accuracy: 0.8639
    Epoch 4060/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3412 - accuracy: 0.8593
    Epoch 4061/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3397 - accuracy: 0.8666
    Epoch 4062/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3421 - accuracy: 0.8675
    Epoch 4063/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3388 - accuracy: 0.8711
    Epoch 4064/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3519 - accuracy: 0.8639
    Epoch 4065/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3433 - accuracy: 0.8684
    Epoch 4066/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3913 - accuracy: 0.8612
    Epoch 4067/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3508 - accuracy: 0.8630
    Epoch 4068/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3381 - accuracy: 0.8657
    Epoch 4069/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3326 - accuracy: 0.8693
    Epoch 4070/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3364 - accuracy: 0.8730
    Epoch 4071/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3590 - accuracy: 0.8603
    Epoch 4072/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3380 - accuracy: 0.8693
    Epoch 4073/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3361 - accuracy: 0.8684
    Epoch 4074/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3338 - accuracy: 0.8711
    Epoch 4075/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3370 - accuracy: 0.8693
    Epoch 4076/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3393 - accuracy: 0.8630
    Epoch 4077/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3483 - accuracy: 0.8648
    Epoch 4078/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3352 - accuracy: 0.8693
    Epoch 4079/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3471 - accuracy: 0.8711
    Epoch 4080/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3375 - accuracy: 0.8721
    Epoch 4081/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3364 - accuracy: 0.8702
    Epoch 4082/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3380 - accuracy: 0.8675
    Epoch 4083/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3565 - accuracy: 0.8639
    Epoch 4084/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3460 - accuracy: 0.8693
    Epoch 4085/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3839 - accuracy: 0.8457
    Epoch 4086/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3449 - accuracy: 0.8603
    Epoch 4087/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3399 - accuracy: 0.8702
    Epoch 4088/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3559 - accuracy: 0.8666
    Epoch 4089/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3350 - accuracy: 0.8666
    Epoch 4090/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3421 - accuracy: 0.8657
    Epoch 4091/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3474 - accuracy: 0.8648
    Epoch 4092/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3381 - accuracy: 0.8702
    Epoch 4093/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3624 - accuracy: 0.8603
    Epoch 4094/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3327 - accuracy: 0.8721
    Epoch 4095/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3489 - accuracy: 0.8648
    Epoch 4096/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3784 - accuracy: 0.8575
    Epoch 4097/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3387 - accuracy: 0.8748
    Epoch 4098/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3331 - accuracy: 0.8748
    Epoch 4099/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3593 - accuracy: 0.8575
    Epoch 4100/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3923 - accuracy: 0.8548
    Epoch 4101/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3400 - accuracy: 0.8693
    Epoch 4102/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3331 - accuracy: 0.8748
    Epoch 4103/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3410 - accuracy: 0.8593
    Epoch 4104/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3688 - accuracy: 0.8575
    Epoch 4105/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3339 - accuracy: 0.8693
    Epoch 4106/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3424 - accuracy: 0.8675
    Epoch 4107/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3443 - accuracy: 0.8557
    Epoch 4108/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8721
    Epoch 4109/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3518 - accuracy: 0.8648
    Epoch 4110/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3902 - accuracy: 0.8521
    Epoch 4111/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3557 - accuracy: 0.8584
    Epoch 4112/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3594 - accuracy: 0.8639
    Epoch 4113/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3450 - accuracy: 0.8702
    Epoch 4114/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3414 - accuracy: 0.8621
    Epoch 4115/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3446 - accuracy: 0.8675
    Epoch 4116/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3393 - accuracy: 0.8675
    Epoch 4117/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3358 - accuracy: 0.8675
    Epoch 4118/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3416 - accuracy: 0.8711
    Epoch 4119/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3315 - accuracy: 0.8721
    Epoch 4120/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3463 - accuracy: 0.8675
    Epoch 4121/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3334 - accuracy: 0.8702
    Epoch 4122/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3438 - accuracy: 0.8575
    Epoch 4123/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3483 - accuracy: 0.8603
    Epoch 4124/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3699 - accuracy: 0.8666
    Epoch 4125/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3640 - accuracy: 0.8566
    Epoch 4126/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3711 - accuracy: 0.8566
    Epoch 4127/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3499 - accuracy: 0.8603
    Epoch 4128/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3325 - accuracy: 0.8711
    Epoch 4129/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3615 - accuracy: 0.8539
    Epoch 4130/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3428 - accuracy: 0.8739
    Epoch 4131/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3423 - accuracy: 0.8630
    Epoch 4132/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3483 - accuracy: 0.8666
    Epoch 4133/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3508 - accuracy: 0.8648
    Epoch 4134/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3391 - accuracy: 0.8657
    Epoch 4135/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3583 - accuracy: 0.8621
    Epoch 4136/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3305 - accuracy: 0.8711
    Epoch 4137/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3515 - accuracy: 0.8693
    Epoch 4138/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3872 - accuracy: 0.8566
    Epoch 4139/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3533 - accuracy: 0.8584
    Epoch 4140/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3339 - accuracy: 0.8757
    Epoch 4141/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3333 - accuracy: 0.8639
    Epoch 4142/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3414 - accuracy: 0.8639
    Epoch 4143/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3465 - accuracy: 0.8657
    Epoch 4144/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3675 - accuracy: 0.8557
    Epoch 4145/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3719 - accuracy: 0.8557
    Epoch 4146/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3563 - accuracy: 0.8684
    Epoch 4147/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3572 - accuracy: 0.8648
    Epoch 4148/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3584 - accuracy: 0.8548
    Epoch 4149/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3352 - accuracy: 0.8693
    Epoch 4150/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3340 - accuracy: 0.8711
    Epoch 4151/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3498 - accuracy: 0.8539
    Epoch 4152/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3361 - accuracy: 0.8675
    Epoch 4153/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3383 - accuracy: 0.8693
    Epoch 4154/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3378 - accuracy: 0.8702
    Epoch 4155/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3396 - accuracy: 0.8684
    Epoch 4156/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3285 - accuracy: 0.8730
    Epoch 4157/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3506 - accuracy: 0.8693
    Epoch 4158/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3570 - accuracy: 0.8702
    Epoch 4159/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3365 - accuracy: 0.8666
    Epoch 4160/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3409 - accuracy: 0.8684
    Epoch 4161/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3349 - accuracy: 0.8757
    Epoch 4162/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3384 - accuracy: 0.8675
    Epoch 4163/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3727 - accuracy: 0.8503
    Epoch 4164/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3467 - accuracy: 0.8675
    Epoch 4165/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3400 - accuracy: 0.8648
    Epoch 4166/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3414 - accuracy: 0.8630
    Epoch 4167/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3470 - accuracy: 0.8684
    Epoch 4168/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3391 - accuracy: 0.8666
    Epoch 4169/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3526 - accuracy: 0.8675
    Epoch 4170/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3321 - accuracy: 0.8711
    Epoch 4171/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3552 - accuracy: 0.8593
    Epoch 4172/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3301 - accuracy: 0.8739
    Epoch 4173/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3426 - accuracy: 0.8721
    Epoch 4174/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3540 - accuracy: 0.8657
    Epoch 4175/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4006 - accuracy: 0.8385
    Epoch 4176/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3340 - accuracy: 0.8757
    Epoch 4177/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3348 - accuracy: 0.8702
    Epoch 4178/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3374 - accuracy: 0.8766
    Epoch 4179/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3353 - accuracy: 0.8739
    Epoch 4180/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3486 - accuracy: 0.8639
    Epoch 4181/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3347 - accuracy: 0.8711
    Epoch 4182/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3500 - accuracy: 0.8566
    Epoch 4183/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3475 - accuracy: 0.8711
    Epoch 4184/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3578 - accuracy: 0.8584
    Epoch 4185/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3618 - accuracy: 0.8593
    Epoch 4186/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3367 - accuracy: 0.8657
    Epoch 4187/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3595 - accuracy: 0.8648
    Epoch 4188/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3484 - accuracy: 0.8593
    Epoch 4189/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3463 - accuracy: 0.8621
    Epoch 4190/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3374 - accuracy: 0.8693
    Epoch 4191/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3588 - accuracy: 0.8575
    Epoch 4192/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3352 - accuracy: 0.8730
    Epoch 4193/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3498 - accuracy: 0.8639
    Epoch 4194/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3518 - accuracy: 0.8584
    Epoch 4195/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3654 - accuracy: 0.8557
    Epoch 4196/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3382 - accuracy: 0.8748
    Epoch 4197/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3396 - accuracy: 0.8702
    Epoch 4198/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3511 - accuracy: 0.8739
    Epoch 4199/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3454 - accuracy: 0.8684
    Epoch 4200/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3742 - accuracy: 0.8548
    Epoch 4201/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3448 - accuracy: 0.8675
    Epoch 4202/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3397 - accuracy: 0.8648
    Epoch 4203/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3431 - accuracy: 0.8648
    Epoch 4204/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3758 - accuracy: 0.8612
    Epoch 4205/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3281 - accuracy: 0.8730
    Epoch 4206/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3443 - accuracy: 0.8702
    Epoch 4207/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3468 - accuracy: 0.8639
    Epoch 4208/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3527 - accuracy: 0.8693
    Epoch 4209/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8739
    Epoch 4210/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3645 - accuracy: 0.8666
    Epoch 4211/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3370 - accuracy: 0.8702
    Epoch 4212/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3381 - accuracy: 0.8766
    Epoch 4213/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3669 - accuracy: 0.8485
    Epoch 4214/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3600 - accuracy: 0.8630
    Epoch 4215/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8711
    Epoch 4216/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3475 - accuracy: 0.8684
    Epoch 4217/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3515 - accuracy: 0.8702
    Epoch 4218/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3360 - accuracy: 0.8793
    Epoch 4219/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3398 - accuracy: 0.8693
    Epoch 4220/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3397 - accuracy: 0.8730
    Epoch 4221/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8730
    Epoch 4222/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8721
    Epoch 4223/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3372 - accuracy: 0.8693
    Epoch 4224/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3394 - accuracy: 0.8702
    Epoch 4225/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3454 - accuracy: 0.8657
    Epoch 4226/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3345 - accuracy: 0.8693
    Epoch 4227/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3441 - accuracy: 0.8648
    Epoch 4228/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3337 - accuracy: 0.8721
    Epoch 4229/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3342 - accuracy: 0.8784
    Epoch 4230/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3308 - accuracy: 0.8711
    Epoch 4231/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3357 - accuracy: 0.8730
    Epoch 4232/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3351 - accuracy: 0.8748
    Epoch 4233/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3417 - accuracy: 0.8702
    Epoch 4234/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3355 - accuracy: 0.8757
    Epoch 4235/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3397 - accuracy: 0.8684
    Epoch 4236/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3297 - accuracy: 0.8793
    Epoch 4237/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3343 - accuracy: 0.8757
    Epoch 4238/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3339 - accuracy: 0.8721
    Epoch 4239/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3467 - accuracy: 0.8721
    Epoch 4240/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3495 - accuracy: 0.8557
    Epoch 4241/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.4215 - accuracy: 0.8394
    Epoch 4242/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3549 - accuracy: 0.8666
    Epoch 4243/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3459 - accuracy: 0.8630
    Epoch 4244/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3895 - accuracy: 0.8376
    Epoch 4245/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3382 - accuracy: 0.8775
    Epoch 4246/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3476 - accuracy: 0.8612
    Epoch 4247/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3623 - accuracy: 0.8575
    Epoch 4248/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3922 - accuracy: 0.8475
    Epoch 4249/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3452 - accuracy: 0.8748
    Epoch 4250/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3626 - accuracy: 0.8603
    Epoch 4251/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3568 - accuracy: 0.8593
    Epoch 4252/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3432 - accuracy: 0.8621
    Epoch 4253/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3396 - accuracy: 0.8675
    Epoch 4254/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8675
    Epoch 4255/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3800 - accuracy: 0.8593
    Epoch 4256/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3369 - accuracy: 0.8721
    Epoch 4257/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3453 - accuracy: 0.8684
    Epoch 4258/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3349 - accuracy: 0.8711
    Epoch 4259/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3437 - accuracy: 0.8648
    Epoch 4260/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3304 - accuracy: 0.8784
    Epoch 4261/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3638 - accuracy: 0.8530
    Epoch 4262/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3348 - accuracy: 0.8739
    Epoch 4263/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3327 - accuracy: 0.8711
    Epoch 4264/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8721
    Epoch 4265/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3318 - accuracy: 0.8793
    Epoch 4266/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3459 - accuracy: 0.8648
    Epoch 4267/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3459 - accuracy: 0.8648
    Epoch 4268/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3291 - accuracy: 0.8748
    Epoch 4269/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3420 - accuracy: 0.8721
    Epoch 4270/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3332 - accuracy: 0.8748
    Epoch 4271/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3613 - accuracy: 0.8711
    Epoch 4272/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3413 - accuracy: 0.8693
    Epoch 4273/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3439 - accuracy: 0.8612
    Epoch 4274/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3470 - accuracy: 0.8675
    Epoch 4275/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3332 - accuracy: 0.8766
    Epoch 4276/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3418 - accuracy: 0.8693
    Epoch 4277/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3374 - accuracy: 0.8657
    Epoch 4278/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3514 - accuracy: 0.8621
    Epoch 4279/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3486 - accuracy: 0.8657
    Epoch 4280/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3432 - accuracy: 0.8711
    Epoch 4281/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3324 - accuracy: 0.8721
    Epoch 4282/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3402 - accuracy: 0.8675
    Epoch 4283/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3581 - accuracy: 0.8666
    Epoch 4284/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3391 - accuracy: 0.8648
    Epoch 4285/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3359 - accuracy: 0.8711
    Epoch 4286/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3403 - accuracy: 0.8702
    Epoch 4287/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3399 - accuracy: 0.8820
    Epoch 4288/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3419 - accuracy: 0.8675
    Epoch 4289/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3511 - accuracy: 0.8630
    Epoch 4290/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3576 - accuracy: 0.8584
    Epoch 4291/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3516 - accuracy: 0.8630
    Epoch 4292/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3379 - accuracy: 0.8675
    Epoch 4293/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3463 - accuracy: 0.8730
    Epoch 4294/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3334 - accuracy: 0.8702
    Epoch 4295/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3418 - accuracy: 0.8730
    Epoch 4296/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3322 - accuracy: 0.8739
    Epoch 4297/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3373 - accuracy: 0.8648
    Epoch 4298/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3451 - accuracy: 0.8693
    Epoch 4299/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3390 - accuracy: 0.8702
    Epoch 4300/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3394 - accuracy: 0.8693
    Epoch 4301/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3914 - accuracy: 0.8566
    Epoch 4302/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3342 - accuracy: 0.8693
    Epoch 4303/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3315 - accuracy: 0.8730
    Epoch 4304/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3505 - accuracy: 0.8666
    Epoch 4305/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3838 - accuracy: 0.8521
    Epoch 4306/6000
    35/35 [==============================] - 0s 530us/step - loss: 0.3406 - accuracy: 0.8666
    Epoch 4307/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3386 - accuracy: 0.8711
    Epoch 4308/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3530 - accuracy: 0.8575
    Epoch 4309/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3475 - accuracy: 0.8648
    Epoch 4310/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3467 - accuracy: 0.8639
    Epoch 4311/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3530 - accuracy: 0.8630
    Epoch 4312/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3386 - accuracy: 0.8721
    Epoch 4313/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3456 - accuracy: 0.8657
    Epoch 4314/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3392 - accuracy: 0.8711
    Epoch 4315/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3585 - accuracy: 0.8684
    Epoch 4316/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3570 - accuracy: 0.8557
    Epoch 4317/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3548 - accuracy: 0.8603
    Epoch 4318/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3353 - accuracy: 0.8757
    Epoch 4319/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3335 - accuracy: 0.8693
    Epoch 4320/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3410 - accuracy: 0.8684
    Epoch 4321/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3479 - accuracy: 0.8684
    Epoch 4322/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3339 - accuracy: 0.8739
    Epoch 4323/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3357 - accuracy: 0.8666
    Epoch 4324/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3413 - accuracy: 0.8630
    Epoch 4325/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3501 - accuracy: 0.8639
    Epoch 4326/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3359 - accuracy: 0.8702
    Epoch 4327/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3344 - accuracy: 0.8666
    Epoch 4328/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3495 - accuracy: 0.8621
    Epoch 4329/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3291 - accuracy: 0.8739
    Epoch 4330/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3472 - accuracy: 0.8603
    Epoch 4331/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3729 - accuracy: 0.8530
    Epoch 4332/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3439 - accuracy: 0.8675
    Epoch 4333/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3467 - accuracy: 0.8630
    Epoch 4334/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3453 - accuracy: 0.8666
    Epoch 4335/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3635 - accuracy: 0.8666
    Epoch 4336/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3474 - accuracy: 0.8693
    Epoch 4337/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3518 - accuracy: 0.8639
    Epoch 4338/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3725 - accuracy: 0.8621
    Epoch 4339/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3334 - accuracy: 0.8748
    Epoch 4340/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3326 - accuracy: 0.8730
    Epoch 4341/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3314 - accuracy: 0.8721
    Epoch 4342/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3353 - accuracy: 0.8730
    Epoch 4343/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3470 - accuracy: 0.8757
    Epoch 4344/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8639
    Epoch 4345/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3304 - accuracy: 0.8784
    Epoch 4346/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8666
    Epoch 4347/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3574 - accuracy: 0.8648
    Epoch 4348/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3393 - accuracy: 0.8630
    Epoch 4349/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3459 - accuracy: 0.8612
    Epoch 4350/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3310 - accuracy: 0.8721
    Epoch 4351/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3360 - accuracy: 0.8666
    Epoch 4352/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3538 - accuracy: 0.8675
    Epoch 4353/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3709 - accuracy: 0.8566
    Epoch 4354/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3365 - accuracy: 0.8748
    Epoch 4355/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3406 - accuracy: 0.8693
    Epoch 4356/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3322 - accuracy: 0.8775
    Epoch 4357/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3569 - accuracy: 0.8612
    Epoch 4358/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8721
    Epoch 4359/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3393 - accuracy: 0.8730
    Epoch 4360/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3444 - accuracy: 0.8702
    Epoch 4361/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8748
    Epoch 4362/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3425 - accuracy: 0.8666
    Epoch 4363/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3408 - accuracy: 0.8739
    Epoch 4364/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3534 - accuracy: 0.8630
    Epoch 4365/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3933 - accuracy: 0.8466
    Epoch 4366/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3351 - accuracy: 0.8739
    Epoch 4367/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3430 - accuracy: 0.8666
    Epoch 4368/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3311 - accuracy: 0.8739
    Epoch 4369/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3510 - accuracy: 0.8593
    Epoch 4370/6000
    35/35 [==============================] - 0s 882us/step - loss: 0.3488 - accuracy: 0.8530
    Epoch 4371/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3385 - accuracy: 0.8684
    Epoch 4372/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3456 - accuracy: 0.8621
    Epoch 4373/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3661 - accuracy: 0.8639
    Epoch 4374/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3490 - accuracy: 0.8675
    Epoch 4375/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3335 - accuracy: 0.8657
    Epoch 4376/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3882 - accuracy: 0.8566
    Epoch 4377/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3782 - accuracy: 0.8612
    Epoch 4378/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3386 - accuracy: 0.8693
    Epoch 4379/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3455 - accuracy: 0.8621
    Epoch 4380/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3413 - accuracy: 0.8630
    Epoch 4381/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3315 - accuracy: 0.8693
    Epoch 4382/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3473 - accuracy: 0.8666
    Epoch 4383/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3324 - accuracy: 0.8711
    Epoch 4384/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3388 - accuracy: 0.8748
    Epoch 4385/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3426 - accuracy: 0.8702
    Epoch 4386/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3343 - accuracy: 0.8693
    Epoch 4387/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3287 - accuracy: 0.8702
    Epoch 4388/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3554 - accuracy: 0.8612
    Epoch 4389/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3491 - accuracy: 0.8711
    Epoch 4390/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3340 - accuracy: 0.8739
    Epoch 4391/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3631 - accuracy: 0.8584
    Epoch 4392/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3338 - accuracy: 0.8639
    Epoch 4393/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3323 - accuracy: 0.8711
    Epoch 4394/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3323 - accuracy: 0.8775
    Epoch 4395/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3426 - accuracy: 0.8748
    Epoch 4396/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3496 - accuracy: 0.8693
    Epoch 4397/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3397 - accuracy: 0.8657
    Epoch 4398/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3664 - accuracy: 0.8603
    Epoch 4399/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3430 - accuracy: 0.8693
    Epoch 4400/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3568 - accuracy: 0.8639
    Epoch 4401/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3516 - accuracy: 0.8693
    Epoch 4402/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3389 - accuracy: 0.8702
    Epoch 4403/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3349 - accuracy: 0.8748
    Epoch 4404/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3393 - accuracy: 0.8739
    Epoch 4405/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3603 - accuracy: 0.8548
    Epoch 4406/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3487 - accuracy: 0.8657
    Epoch 4407/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3518 - accuracy: 0.8711
    Epoch 4408/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3450 - accuracy: 0.8711
    Epoch 4409/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3439 - accuracy: 0.8639
    Epoch 4410/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3464 - accuracy: 0.8675
    Epoch 4411/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3470 - accuracy: 0.8748
    Epoch 4412/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3571 - accuracy: 0.8711
    Epoch 4413/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3354 - accuracy: 0.8757
    Epoch 4414/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3333 - accuracy: 0.8721
    Epoch 4415/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3366 - accuracy: 0.8739
    Epoch 4416/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3321 - accuracy: 0.8721
    Epoch 4417/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3463 - accuracy: 0.8693
    Epoch 4418/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3442 - accuracy: 0.8721
    Epoch 4419/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3398 - accuracy: 0.8639
    Epoch 4420/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3376 - accuracy: 0.8612
    Epoch 4421/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3512 - accuracy: 0.8603
    Epoch 4422/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3400 - accuracy: 0.8684
    Epoch 4423/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3293 - accuracy: 0.8766
    Epoch 4424/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3345 - accuracy: 0.8721
    Epoch 4425/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3427 - accuracy: 0.8748
    Epoch 4426/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3429 - accuracy: 0.8666
    Epoch 4427/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3324 - accuracy: 0.8684
    Epoch 4428/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3447 - accuracy: 0.8612
    Epoch 4429/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3378 - accuracy: 0.8739
    Epoch 4430/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3476 - accuracy: 0.8693
    Epoch 4431/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3535 - accuracy: 0.8721
    Epoch 4432/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3544 - accuracy: 0.8557
    Epoch 4433/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3386 - accuracy: 0.8711
    Epoch 4434/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3315 - accuracy: 0.8702
    Epoch 4435/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3371 - accuracy: 0.8684
    Epoch 4436/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3386 - accuracy: 0.8693
    Epoch 4437/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3419 - accuracy: 0.8702
    Epoch 4438/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3410 - accuracy: 0.8675
    Epoch 4439/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3295 - accuracy: 0.8721
    Epoch 4440/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3581 - accuracy: 0.8557
    Epoch 4441/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3587 - accuracy: 0.8757
    Epoch 4442/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3422 - accuracy: 0.8666
    Epoch 4443/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3455 - accuracy: 0.8739
    Epoch 4444/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3568 - accuracy: 0.8593
    Epoch 4445/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3705 - accuracy: 0.8612
    Epoch 4446/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3365 - accuracy: 0.8675
    Epoch 4447/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3589 - accuracy: 0.8575
    Epoch 4448/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3306 - accuracy: 0.8721
    Epoch 4449/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3466 - accuracy: 0.8675
    Epoch 4450/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3424 - accuracy: 0.8684
    Epoch 4451/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3590 - accuracy: 0.8711
    Epoch 4452/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3459 - accuracy: 0.8648
    Epoch 4453/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3687 - accuracy: 0.8521
    Epoch 4454/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3349 - accuracy: 0.8684
    Epoch 4455/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3371 - accuracy: 0.8684
    Epoch 4456/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3472 - accuracy: 0.8621
    Epoch 4457/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3340 - accuracy: 0.8648
    Epoch 4458/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3354 - accuracy: 0.8721
    Epoch 4459/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3635 - accuracy: 0.8557
    Epoch 4460/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3557 - accuracy: 0.8584
    Epoch 4461/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3531 - accuracy: 0.8630
    Epoch 4462/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3332 - accuracy: 0.8757
    Epoch 4463/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3396 - accuracy: 0.8648
    Epoch 4464/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3285 - accuracy: 0.8748
    Epoch 4465/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3493 - accuracy: 0.8603
    Epoch 4466/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3866 - accuracy: 0.8603
    Epoch 4467/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3345 - accuracy: 0.8739
    Epoch 4468/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3412 - accuracy: 0.8684
    Epoch 4469/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3446 - accuracy: 0.8675
    Epoch 4470/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3460 - accuracy: 0.8721
    Epoch 4471/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3395 - accuracy: 0.8657
    Epoch 4472/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3461 - accuracy: 0.8666
    Epoch 4473/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3550 - accuracy: 0.8657
    Epoch 4474/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3366 - accuracy: 0.8721
    Epoch 4475/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3320 - accuracy: 0.8721
    Epoch 4476/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3322 - accuracy: 0.8711
    Epoch 4477/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3398 - accuracy: 0.8711
    Epoch 4478/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3325 - accuracy: 0.8675
    Epoch 4479/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3527 - accuracy: 0.8648
    Epoch 4480/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3300 - accuracy: 0.8721
    Epoch 4481/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3444 - accuracy: 0.8675
    Epoch 4482/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3347 - accuracy: 0.8693
    Epoch 4483/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3348 - accuracy: 0.8702
    Epoch 4484/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3357 - accuracy: 0.8784
    Epoch 4485/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3303 - accuracy: 0.8748
    Epoch 4486/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3506 - accuracy: 0.8630
    Epoch 4487/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3311 - accuracy: 0.8684
    Epoch 4488/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3462 - accuracy: 0.8702
    Epoch 4489/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3311 - accuracy: 0.8739
    Epoch 4490/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3280 - accuracy: 0.8811
    Epoch 4491/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3286 - accuracy: 0.8766
    Epoch 4492/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3419 - accuracy: 0.8666
    Epoch 4493/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3595 - accuracy: 0.8566
    Epoch 4494/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3371 - accuracy: 0.8684
    Epoch 4495/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3800 - accuracy: 0.8439
    Epoch 4496/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3417 - accuracy: 0.8684
    Epoch 4497/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3478 - accuracy: 0.8648
    Epoch 4498/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3547 - accuracy: 0.8630
    Epoch 4499/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3557 - accuracy: 0.8557
    Epoch 4500/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3350 - accuracy: 0.8684
    Epoch 4501/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3588 - accuracy: 0.8603
    Epoch 4502/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3656 - accuracy: 0.8630
    Epoch 4503/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3431 - accuracy: 0.8657
    Epoch 4504/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3432 - accuracy: 0.8657
    Epoch 4505/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3535 - accuracy: 0.8693
    Epoch 4506/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3362 - accuracy: 0.8739
    Epoch 4507/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3354 - accuracy: 0.8666
    Epoch 4508/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3360 - accuracy: 0.8693
    Epoch 4509/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3361 - accuracy: 0.8721
    Epoch 4510/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3320 - accuracy: 0.8721
    Epoch 4511/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3424 - accuracy: 0.8757
    Epoch 4512/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3327 - accuracy: 0.8721
    Epoch 4513/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4028 - accuracy: 0.8421
    Epoch 4514/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3766 - accuracy: 0.8584
    Epoch 4515/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3624 - accuracy: 0.8630
    Epoch 4516/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3336 - accuracy: 0.8575
    Epoch 4517/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3466 - accuracy: 0.8666
    Epoch 4518/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3728 - accuracy: 0.8693
    Epoch 4519/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3341 - accuracy: 0.8757
    Epoch 4520/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3441 - accuracy: 0.8648
    Epoch 4521/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3404 - accuracy: 0.8711
    Epoch 4522/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3417 - accuracy: 0.8639
    Epoch 4523/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3299 - accuracy: 0.8675
    Epoch 4524/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3480 - accuracy: 0.8693
    Epoch 4525/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3333 - accuracy: 0.8684
    Epoch 4526/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3333 - accuracy: 0.8730
    Epoch 4527/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8721
    Epoch 4528/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3322 - accuracy: 0.8721
    Epoch 4529/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3882 - accuracy: 0.8530
    Epoch 4530/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3402 - accuracy: 0.8648
    Epoch 4531/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3323 - accuracy: 0.8748
    Epoch 4532/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3370 - accuracy: 0.8721
    Epoch 4533/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3318 - accuracy: 0.8766
    Epoch 4534/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3333 - accuracy: 0.8748
    Epoch 4535/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3327 - accuracy: 0.8711
    Epoch 4536/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3533 - accuracy: 0.8621
    Epoch 4537/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3400 - accuracy: 0.8639
    Epoch 4538/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3396 - accuracy: 0.8639
    Epoch 4539/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3351 - accuracy: 0.8711
    Epoch 4540/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3356 - accuracy: 0.8675
    Epoch 4541/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3414 - accuracy: 0.8730
    Epoch 4542/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3675 - accuracy: 0.8593
    Epoch 4543/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3479 - accuracy: 0.8630
    Epoch 4544/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3433 - accuracy: 0.8684
    Epoch 4545/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3588 - accuracy: 0.8557
    Epoch 4546/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3349 - accuracy: 0.8702
    Epoch 4547/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3340 - accuracy: 0.8711
    Epoch 4548/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3498 - accuracy: 0.8639
    Epoch 4549/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3515 - accuracy: 0.8730
    Epoch 4550/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3344 - accuracy: 0.8775
    Epoch 4551/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3422 - accuracy: 0.8657
    Epoch 4552/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3375 - accuracy: 0.8748
    Epoch 4553/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3336 - accuracy: 0.8784
    Epoch 4554/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3478 - accuracy: 0.8630
    Epoch 4555/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3530 - accuracy: 0.8648
    Epoch 4556/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3443 - accuracy: 0.8621
    Epoch 4557/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3394 - accuracy: 0.8657
    Epoch 4558/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3796 - accuracy: 0.8530
    Epoch 4559/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3432 - accuracy: 0.8639
    Epoch 4560/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3400 - accuracy: 0.8639
    Epoch 4561/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3410 - accuracy: 0.8684
    Epoch 4562/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3329 - accuracy: 0.8784
    Epoch 4563/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3384 - accuracy: 0.8702
    Epoch 4564/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3462 - accuracy: 0.8657
    Epoch 4565/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3313 - accuracy: 0.8757
    Epoch 4566/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3415 - accuracy: 0.8593
    Epoch 4567/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3410 - accuracy: 0.8739
    Epoch 4568/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3410 - accuracy: 0.8639
    Epoch 4569/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3431 - accuracy: 0.8684
    Epoch 4570/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3401 - accuracy: 0.8739
    Epoch 4571/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3570 - accuracy: 0.8612
    Epoch 4572/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3569 - accuracy: 0.8575
    Epoch 4573/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3879 - accuracy: 0.8521
    Epoch 4574/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3384 - accuracy: 0.8711
    Epoch 4575/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8702
    Epoch 4576/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3329 - accuracy: 0.8711
    Epoch 4577/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3548 - accuracy: 0.8621
    Epoch 4578/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3458 - accuracy: 0.8666
    Epoch 4579/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3597 - accuracy: 0.8675
    Epoch 4580/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3365 - accuracy: 0.8702
    Epoch 4581/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3403 - accuracy: 0.8675
    Epoch 4582/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8684
    Epoch 4583/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3564 - accuracy: 0.8593
    Epoch 4584/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3424 - accuracy: 0.8630
    Epoch 4585/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3403 - accuracy: 0.8684
    Epoch 4586/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3320 - accuracy: 0.8757
    Epoch 4587/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3367 - accuracy: 0.8693
    Epoch 4588/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3483 - accuracy: 0.8702
    Epoch 4589/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3575 - accuracy: 0.8675
    Epoch 4590/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3450 - accuracy: 0.8621
    Epoch 4591/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3305 - accuracy: 0.8721
    Epoch 4592/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3362 - accuracy: 0.8675
    Epoch 4593/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3583 - accuracy: 0.8593
    Epoch 4594/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3306 - accuracy: 0.8802
    Epoch 4595/6000
    35/35 [==============================] - 0s 530us/step - loss: 0.3312 - accuracy: 0.8775
    Epoch 4596/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3386 - accuracy: 0.8775
    Epoch 4597/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3414 - accuracy: 0.8739
    Epoch 4598/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3382 - accuracy: 0.8693
    Epoch 4599/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3499 - accuracy: 0.8702
    Epoch 4600/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3473 - accuracy: 0.8693
    Epoch 4601/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3441 - accuracy: 0.8639
    Epoch 4602/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3454 - accuracy: 0.8675
    Epoch 4603/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3556 - accuracy: 0.8739
    Epoch 4604/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3381 - accuracy: 0.8648
    Epoch 4605/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3348 - accuracy: 0.8721
    Epoch 4606/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3306 - accuracy: 0.8784
    Epoch 4607/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3489 - accuracy: 0.8593
    Epoch 4608/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3363 - accuracy: 0.8711
    Epoch 4609/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3371 - accuracy: 0.8721
    Epoch 4610/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3420 - accuracy: 0.8711
    Epoch 4611/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3265 - accuracy: 0.8820
    Epoch 4612/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3438 - accuracy: 0.8693
    Epoch 4613/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3319 - accuracy: 0.8739
    Epoch 4614/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3414 - accuracy: 0.8639
    Epoch 4615/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3279 - accuracy: 0.8811
    Epoch 4616/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3497 - accuracy: 0.8621
    Epoch 4617/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3584 - accuracy: 0.8593
    Epoch 4618/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3317 - accuracy: 0.8702
    Epoch 4619/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3353 - accuracy: 0.8657
    Epoch 4620/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3648 - accuracy: 0.8494
    Epoch 4621/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3373 - accuracy: 0.8711
    Epoch 4622/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3460 - accuracy: 0.8666
    Epoch 4623/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3498 - accuracy: 0.8621
    Epoch 4624/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3406 - accuracy: 0.8711
    Epoch 4625/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3289 - accuracy: 0.8721
    Epoch 4626/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3588 - accuracy: 0.8584
    Epoch 4627/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3303 - accuracy: 0.8820
    Epoch 4628/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3399 - accuracy: 0.8702
    Epoch 4629/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3344 - accuracy: 0.8793
    Epoch 4630/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3415 - accuracy: 0.8730
    Epoch 4631/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3463 - accuracy: 0.8748
    Epoch 4632/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3398 - accuracy: 0.8702
    Epoch 4633/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3402 - accuracy: 0.8739
    Epoch 4634/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3381 - accuracy: 0.8721
    Epoch 4635/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3340 - accuracy: 0.8739
    Epoch 4636/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3462 - accuracy: 0.8730
    Epoch 4637/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3555 - accuracy: 0.8684
    Epoch 4638/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3267 - accuracy: 0.8739
    Epoch 4639/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3375 - accuracy: 0.8711
    Epoch 4640/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3315 - accuracy: 0.8802
    Epoch 4641/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3565 - accuracy: 0.8630
    Epoch 4642/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.4605 - accuracy: 0.8303
    Epoch 4643/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3701 - accuracy: 0.8630
    Epoch 4644/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8693
    Epoch 4645/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3430 - accuracy: 0.8630
    Epoch 4646/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3397 - accuracy: 0.8721
    Epoch 4647/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3375 - accuracy: 0.8702
    Epoch 4648/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3314 - accuracy: 0.8757
    Epoch 4649/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3446 - accuracy: 0.8784
    Epoch 4650/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3495 - accuracy: 0.8639
    Epoch 4651/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3399 - accuracy: 0.8739
    Epoch 4652/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3361 - accuracy: 0.8675
    Epoch 4653/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3561 - accuracy: 0.8539
    Epoch 4654/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3436 - accuracy: 0.8721
    Epoch 4655/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3513 - accuracy: 0.8702
    Epoch 4656/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3533 - accuracy: 0.8711
    Epoch 4657/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3438 - accuracy: 0.8702
    Epoch 4658/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3316 - accuracy: 0.8693
    Epoch 4659/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3319 - accuracy: 0.8693
    Epoch 4660/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3570 - accuracy: 0.8702
    Epoch 4661/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3493 - accuracy: 0.8721
    Epoch 4662/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3390 - accuracy: 0.8684
    Epoch 4663/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3406 - accuracy: 0.8711
    Epoch 4664/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3379 - accuracy: 0.8630
    Epoch 4665/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3559 - accuracy: 0.8566
    Epoch 4666/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3369 - accuracy: 0.8675
    Epoch 4667/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3280 - accuracy: 0.8739
    Epoch 4668/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3342 - accuracy: 0.8675
    Epoch 4669/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3500 - accuracy: 0.8530
    Epoch 4670/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3702 - accuracy: 0.8648
    Epoch 4671/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3292 - accuracy: 0.8693
    Epoch 4672/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3392 - accuracy: 0.8711
    Epoch 4673/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3299 - accuracy: 0.8730
    Epoch 4674/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3526 - accuracy: 0.8621
    Epoch 4675/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3416 - accuracy: 0.8657
    Epoch 4676/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3279 - accuracy: 0.8838
    Epoch 4677/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3443 - accuracy: 0.8603
    Epoch 4678/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3494 - accuracy: 0.8721
    Epoch 4679/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3837 - accuracy: 0.8548
    Epoch 4680/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3376 - accuracy: 0.8702
    Epoch 4681/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3402 - accuracy: 0.8730
    Epoch 4682/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3361 - accuracy: 0.8730
    Epoch 4683/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3426 - accuracy: 0.8684
    Epoch 4684/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3411 - accuracy: 0.8657
    Epoch 4685/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3329 - accuracy: 0.8739
    Epoch 4686/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3488 - accuracy: 0.8702
    Epoch 4687/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3495 - accuracy: 0.8711
    Epoch 4688/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3374 - accuracy: 0.8730
    Epoch 4689/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3324 - accuracy: 0.8766
    Epoch 4690/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3413 - accuracy: 0.8784
    Epoch 4691/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8711
    Epoch 4692/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3368 - accuracy: 0.8648
    Epoch 4693/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3389 - accuracy: 0.8702
    Epoch 4694/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3438 - accuracy: 0.8739
    Epoch 4695/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3330 - accuracy: 0.8721
    Epoch 4696/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3362 - accuracy: 0.8711
    Epoch 4697/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3532 - accuracy: 0.8693
    Epoch 4698/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3403 - accuracy: 0.8675
    Epoch 4699/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3378 - accuracy: 0.8702
    Epoch 4700/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3293 - accuracy: 0.8757
    Epoch 4701/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3367 - accuracy: 0.8702
    Epoch 4702/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8684
    Epoch 4703/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3289 - accuracy: 0.8784
    Epoch 4704/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3366 - accuracy: 0.8711
    Epoch 4705/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3328 - accuracy: 0.8702
    Epoch 4706/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3611 - accuracy: 0.8548
    Epoch 4707/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3435 - accuracy: 0.8739
    Epoch 4708/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3398 - accuracy: 0.8684
    Epoch 4709/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3415 - accuracy: 0.8711
    Epoch 4710/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3640 - accuracy: 0.8557
    Epoch 4711/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3367 - accuracy: 0.8748
    Epoch 4712/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3371 - accuracy: 0.8675
    Epoch 4713/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3480 - accuracy: 0.8666
    Epoch 4714/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3521 - accuracy: 0.8621
    Epoch 4715/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3296 - accuracy: 0.8784
    Epoch 4716/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3331 - accuracy: 0.8711
    Epoch 4717/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3458 - accuracy: 0.8684
    Epoch 4718/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3396 - accuracy: 0.8748
    Epoch 4719/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3373 - accuracy: 0.8748
    Epoch 4720/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3427 - accuracy: 0.8775
    Epoch 4721/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3508 - accuracy: 0.8639
    Epoch 4722/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3395 - accuracy: 0.8757
    Epoch 4723/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3311 - accuracy: 0.8748
    Epoch 4724/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3360 - accuracy: 0.8711
    Epoch 4725/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3479 - accuracy: 0.8721
    Epoch 4726/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3688 - accuracy: 0.8566
    Epoch 4727/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3304 - accuracy: 0.8748
    Epoch 4728/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3357 - accuracy: 0.8757
    Epoch 4729/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3390 - accuracy: 0.8748
    Epoch 4730/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3414 - accuracy: 0.8711
    Epoch 4731/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3460 - accuracy: 0.8639
    Epoch 4732/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3784 - accuracy: 0.8584
    Epoch 4733/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3328 - accuracy: 0.8793
    Epoch 4734/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3551 - accuracy: 0.8684
    Epoch 4735/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3376 - accuracy: 0.8711
    Epoch 4736/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3394 - accuracy: 0.8693
    Epoch 4737/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3364 - accuracy: 0.8711
    Epoch 4738/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3350 - accuracy: 0.8739
    Epoch 4739/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3440 - accuracy: 0.8684
    Epoch 4740/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3409 - accuracy: 0.8730
    Epoch 4741/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3462 - accuracy: 0.8530
    Epoch 4742/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3467 - accuracy: 0.8748
    Epoch 4743/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3379 - accuracy: 0.8693
    Epoch 4744/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3451 - accuracy: 0.8657
    Epoch 4745/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3279 - accuracy: 0.8838
    Epoch 4746/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3645 - accuracy: 0.8548
    Epoch 4747/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3626 - accuracy: 0.8593
    Epoch 4748/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8721
    Epoch 4749/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3314 - accuracy: 0.8748
    Epoch 4750/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3319 - accuracy: 0.8648
    Epoch 4751/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3310 - accuracy: 0.8775
    Epoch 4752/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3278 - accuracy: 0.8757
    Epoch 4753/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3372 - accuracy: 0.8721
    Epoch 4754/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3466 - accuracy: 0.8693
    Epoch 4755/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3321 - accuracy: 0.8711
    Epoch 4756/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3252 - accuracy: 0.8793
    Epoch 4757/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3540 - accuracy: 0.8657
    Epoch 4758/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3377 - accuracy: 0.8639
    Epoch 4759/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3681 - accuracy: 0.8475
    Epoch 4760/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3357 - accuracy: 0.8711
    Epoch 4761/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3382 - accuracy: 0.8666
    Epoch 4762/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3320 - accuracy: 0.8675
    Epoch 4763/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3477 - accuracy: 0.8648
    Epoch 4764/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3294 - accuracy: 0.8748
    Epoch 4765/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3304 - accuracy: 0.8784
    Epoch 4766/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3461 - accuracy: 0.8693
    Epoch 4767/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3605 - accuracy: 0.8494
    Epoch 4768/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3347 - accuracy: 0.8739
    Epoch 4769/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3303 - accuracy: 0.8739
    Epoch 4770/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3343 - accuracy: 0.8739
    Epoch 4771/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3446 - accuracy: 0.8693
    Epoch 4772/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3420 - accuracy: 0.8730
    Epoch 4773/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3648 - accuracy: 0.8593
    Epoch 4774/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3540 - accuracy: 0.8684
    Epoch 4775/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3340 - accuracy: 0.8775
    Epoch 4776/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3345 - accuracy: 0.8730
    Epoch 4777/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3461 - accuracy: 0.8666
    Epoch 4778/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3246 - accuracy: 0.8838
    Epoch 4779/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3474 - accuracy: 0.8693
    Epoch 4780/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3335 - accuracy: 0.8739
    Epoch 4781/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3465 - accuracy: 0.8639
    Epoch 4782/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3546 - accuracy: 0.8603
    Epoch 4783/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3330 - accuracy: 0.8757
    Epoch 4784/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3301 - accuracy: 0.8757
    Epoch 4785/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3306 - accuracy: 0.8702
    Epoch 4786/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3271 - accuracy: 0.8739
    Epoch 4787/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3301 - accuracy: 0.8757
    Epoch 4788/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3421 - accuracy: 0.8630
    Epoch 4789/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3703 - accuracy: 0.8548
    Epoch 4790/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3493 - accuracy: 0.8639
    Epoch 4791/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3334 - accuracy: 0.8739
    Epoch 4792/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3269 - accuracy: 0.8666
    Epoch 4793/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3305 - accuracy: 0.8730
    Epoch 4794/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3326 - accuracy: 0.8739
    Epoch 4795/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3384 - accuracy: 0.8630
    Epoch 4796/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3540 - accuracy: 0.8657
    Epoch 4797/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3350 - accuracy: 0.8693
    Epoch 4798/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3369 - accuracy: 0.8757
    Epoch 4799/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3479 - accuracy: 0.8639
    Epoch 4800/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3786 - accuracy: 0.8603
    Epoch 4801/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3327 - accuracy: 0.8793
    Epoch 4802/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3441 - accuracy: 0.8675
    Epoch 4803/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3272 - accuracy: 0.8748
    Epoch 4804/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3359 - accuracy: 0.8784
    Epoch 4805/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3407 - accuracy: 0.8648
    Epoch 4806/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3388 - accuracy: 0.8702
    Epoch 4807/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3346 - accuracy: 0.8702
    Epoch 4808/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8666
    Epoch 4809/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3301 - accuracy: 0.8775
    Epoch 4810/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3255 - accuracy: 0.8757
    Epoch 4811/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3429 - accuracy: 0.8702
    Epoch 4812/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3531 - accuracy: 0.8657
    Epoch 4813/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3455 - accuracy: 0.8639
    Epoch 4814/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3462 - accuracy: 0.8739
    Epoch 4815/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3351 - accuracy: 0.8721
    Epoch 4816/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3304 - accuracy: 0.8757
    Epoch 4817/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3455 - accuracy: 0.8657
    Epoch 4818/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3447 - accuracy: 0.8702
    Epoch 4819/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3319 - accuracy: 0.8721
    Epoch 4820/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3437 - accuracy: 0.8721
    Epoch 4821/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3366 - accuracy: 0.8675
    Epoch 4822/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3713 - accuracy: 0.8657
    Epoch 4823/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3527 - accuracy: 0.8630
    Epoch 4824/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3524 - accuracy: 0.8766
    Epoch 4825/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3459 - accuracy: 0.8702
    Epoch 4826/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3558 - accuracy: 0.8648
    Epoch 4827/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3301 - accuracy: 0.8748
    Epoch 4828/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3857 - accuracy: 0.8539
    Epoch 4829/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3353 - accuracy: 0.8730
    Epoch 4830/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3355 - accuracy: 0.8793
    Epoch 4831/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3421 - accuracy: 0.8730
    Epoch 4832/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3372 - accuracy: 0.8711
    Epoch 4833/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3294 - accuracy: 0.8730
    Epoch 4834/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3487 - accuracy: 0.8657
    Epoch 4835/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3489 - accuracy: 0.8684
    Epoch 4836/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3384 - accuracy: 0.8693
    Epoch 4837/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3472 - accuracy: 0.8711
    Epoch 4838/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3501 - accuracy: 0.8666
    Epoch 4839/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3398 - accuracy: 0.8648
    Epoch 4840/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3649 - accuracy: 0.8621
    Epoch 4841/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3348 - accuracy: 0.8684
    Epoch 4842/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3399 - accuracy: 0.8693
    Epoch 4843/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3359 - accuracy: 0.8730
    Epoch 4844/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3299 - accuracy: 0.8775
    Epoch 4845/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3658 - accuracy: 0.8584
    Epoch 4846/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3353 - accuracy: 0.8730
    Epoch 4847/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3276 - accuracy: 0.8784
    Epoch 4848/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3342 - accuracy: 0.8648
    Epoch 4849/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3626 - accuracy: 0.8612
    Epoch 4850/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3385 - accuracy: 0.8693
    Epoch 4851/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3462 - accuracy: 0.8630
    Epoch 4852/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3356 - accuracy: 0.8748
    Epoch 4853/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3396 - accuracy: 0.8811
    Epoch 4854/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3450 - accuracy: 0.8666
    Epoch 4855/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3532 - accuracy: 0.8648
    Epoch 4856/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3545 - accuracy: 0.8675
    Epoch 4857/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3445 - accuracy: 0.8721
    Epoch 4858/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3541 - accuracy: 0.8593
    Epoch 4859/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3538 - accuracy: 0.8639
    Epoch 4860/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3555 - accuracy: 0.8557
    Epoch 4861/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3466 - accuracy: 0.8639
    Epoch 4862/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3397 - accuracy: 0.8675
    Epoch 4863/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3323 - accuracy: 0.8730
    Epoch 4864/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3440 - accuracy: 0.8657
    Epoch 4865/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3349 - accuracy: 0.8702
    Epoch 4866/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3355 - accuracy: 0.8730
    Epoch 4867/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3373 - accuracy: 0.8684
    Epoch 4868/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3496 - accuracy: 0.8612
    Epoch 4869/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3398 - accuracy: 0.8684
    Epoch 4870/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3340 - accuracy: 0.8775
    Epoch 4871/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3362 - accuracy: 0.8766
    Epoch 4872/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3423 - accuracy: 0.8693
    Epoch 4873/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8648
    Epoch 4874/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3868 - accuracy: 0.8485
    Epoch 4875/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3918 - accuracy: 0.8503
    Epoch 4876/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3318 - accuracy: 0.8784
    Epoch 4877/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3335 - accuracy: 0.8721
    Epoch 4878/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3369 - accuracy: 0.8666
    Epoch 4879/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3284 - accuracy: 0.8693
    Epoch 4880/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3369 - accuracy: 0.8684
    Epoch 4881/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3429 - accuracy: 0.8675
    Epoch 4882/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3463 - accuracy: 0.8757
    Epoch 4883/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3498 - accuracy: 0.8648
    Epoch 4884/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3353 - accuracy: 0.8711
    Epoch 4885/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3456 - accuracy: 0.8748
    Epoch 4886/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3522 - accuracy: 0.8666
    Epoch 4887/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3681 - accuracy: 0.8557
    Epoch 4888/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3404 - accuracy: 0.8648
    Epoch 4889/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3484 - accuracy: 0.8630
    Epoch 4890/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3365 - accuracy: 0.8721
    Epoch 4891/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3279 - accuracy: 0.8793
    Epoch 4892/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3369 - accuracy: 0.8730
    Epoch 4893/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3308 - accuracy: 0.8748
    Epoch 4894/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3349 - accuracy: 0.8675
    Epoch 4895/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3331 - accuracy: 0.8757
    Epoch 4896/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3454 - accuracy: 0.8648
    Epoch 4897/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3486 - accuracy: 0.8675
    Epoch 4898/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3475 - accuracy: 0.8639
    Epoch 4899/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3544 - accuracy: 0.8630
    Epoch 4900/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3424 - accuracy: 0.8739
    Epoch 4901/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3606 - accuracy: 0.8593
    Epoch 4902/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3469 - accuracy: 0.8648
    Epoch 4903/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3390 - accuracy: 0.8711
    Epoch 4904/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3316 - accuracy: 0.8730
    Epoch 4905/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3364 - accuracy: 0.8748
    Epoch 4906/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3350 - accuracy: 0.8739
    Epoch 4907/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3350 - accuracy: 0.8702
    Epoch 4908/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3336 - accuracy: 0.8793
    Epoch 4909/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3370 - accuracy: 0.8757
    Epoch 4910/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3343 - accuracy: 0.8711
    Epoch 4911/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3658 - accuracy: 0.8575
    Epoch 4912/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3429 - accuracy: 0.8721
    Epoch 4913/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3494 - accuracy: 0.8603
    Epoch 4914/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3399 - accuracy: 0.8693
    Epoch 4915/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3313 - accuracy: 0.8693
    Epoch 4916/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3286 - accuracy: 0.8739
    Epoch 4917/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3351 - accuracy: 0.8702
    Epoch 4918/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3342 - accuracy: 0.8702
    Epoch 4919/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3517 - accuracy: 0.8684
    Epoch 4920/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3501 - accuracy: 0.8603
    Epoch 4921/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3662 - accuracy: 0.8584
    Epoch 4922/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3483 - accuracy: 0.8639
    Epoch 4923/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3525 - accuracy: 0.8630
    Epoch 4924/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3445 - accuracy: 0.8648
    Epoch 4925/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3587 - accuracy: 0.8639
    Epoch 4926/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3599 - accuracy: 0.8702
    Epoch 4927/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3330 - accuracy: 0.8693
    Epoch 4928/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3301 - accuracy: 0.8730
    Epoch 4929/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3303 - accuracy: 0.8739
    Epoch 4930/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3313 - accuracy: 0.8757
    Epoch 4931/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3254 - accuracy: 0.8784
    Epoch 4932/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3349 - accuracy: 0.8721
    Epoch 4933/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3447 - accuracy: 0.8675
    Epoch 4934/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3329 - accuracy: 0.8748
    Epoch 4935/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3365 - accuracy: 0.8820
    Epoch 4936/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3400 - accuracy: 0.8711
    Epoch 4937/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3399 - accuracy: 0.8675
    Epoch 4938/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3279 - accuracy: 0.8775
    Epoch 4939/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3425 - accuracy: 0.8684
    Epoch 4940/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3279 - accuracy: 0.8748
    Epoch 4941/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3307 - accuracy: 0.8693
    Epoch 4942/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3543 - accuracy: 0.8584
    Epoch 4943/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3291 - accuracy: 0.8711
    Epoch 4944/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3753 - accuracy: 0.8512
    Epoch 4945/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3324 - accuracy: 0.8693
    Epoch 4946/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3293 - accuracy: 0.8693
    Epoch 4947/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3312 - accuracy: 0.8702
    Epoch 4948/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3511 - accuracy: 0.8603
    Epoch 4949/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3327 - accuracy: 0.8693
    Epoch 4950/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3390 - accuracy: 0.8748
    Epoch 4951/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3374 - accuracy: 0.8702
    Epoch 4952/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3515 - accuracy: 0.8603
    Epoch 4953/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3575 - accuracy: 0.8566
    Epoch 4954/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3337 - accuracy: 0.8693
    Epoch 4955/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3411 - accuracy: 0.8675
    Epoch 4956/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3329 - accuracy: 0.8711
    Epoch 4957/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8702
    Epoch 4958/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3383 - accuracy: 0.8675
    Epoch 4959/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3503 - accuracy: 0.8693
    Epoch 4960/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3307 - accuracy: 0.8793
    Epoch 4961/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3327 - accuracy: 0.8648
    Epoch 4962/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3488 - accuracy: 0.8657
    Epoch 4963/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3602 - accuracy: 0.8566
    Epoch 4964/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3384 - accuracy: 0.8721
    Epoch 4965/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3279 - accuracy: 0.8757
    Epoch 4966/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3448 - accuracy: 0.8621
    Epoch 4967/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3343 - accuracy: 0.8748
    Epoch 4968/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3395 - accuracy: 0.8730
    Epoch 4969/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3298 - accuracy: 0.8730
    Epoch 4970/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3288 - accuracy: 0.8757
    Epoch 4971/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3280 - accuracy: 0.8748
    Epoch 4972/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3376 - accuracy: 0.8775
    Epoch 4973/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3711 - accuracy: 0.8548
    Epoch 4974/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3449 - accuracy: 0.8757
    Epoch 4975/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3340 - accuracy: 0.8748
    Epoch 4976/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3335 - accuracy: 0.8802
    Epoch 4977/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3394 - accuracy: 0.8702
    Epoch 4978/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3481 - accuracy: 0.8675
    Epoch 4979/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3603 - accuracy: 0.8539
    Epoch 4980/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3471 - accuracy: 0.8603
    Epoch 4981/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3348 - accuracy: 0.8748
    Epoch 4982/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3374 - accuracy: 0.8711
    Epoch 4983/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3449 - accuracy: 0.8621
    Epoch 4984/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3406 - accuracy: 0.8730
    Epoch 4985/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3639 - accuracy: 0.8566
    Epoch 4986/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3493 - accuracy: 0.8711
    Epoch 4987/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3340 - accuracy: 0.8739
    Epoch 4988/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3315 - accuracy: 0.8730
    Epoch 4989/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3375 - accuracy: 0.8739
    Epoch 4990/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3565 - accuracy: 0.8584
    Epoch 4991/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3450 - accuracy: 0.8675
    Epoch 4992/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3361 - accuracy: 0.8648
    Epoch 4993/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3371 - accuracy: 0.8711
    Epoch 4994/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3326 - accuracy: 0.8757
    Epoch 4995/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3465 - accuracy: 0.8684
    Epoch 4996/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3303 - accuracy: 0.8775
    Epoch 4997/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3270 - accuracy: 0.8775
    Epoch 4998/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3651 - accuracy: 0.8530
    Epoch 4999/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3457 - accuracy: 0.8666
    Epoch 5000/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3537 - accuracy: 0.8739
    Epoch 5001/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3352 - accuracy: 0.8721
    Epoch 5002/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3461 - accuracy: 0.8639
    Epoch 5003/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3435 - accuracy: 0.8702
    Epoch 5004/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3296 - accuracy: 0.8757
    Epoch 5005/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3679 - accuracy: 0.8566
    Epoch 5006/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3338 - accuracy: 0.8702
    Epoch 5007/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3331 - accuracy: 0.8739
    Epoch 5008/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3679 - accuracy: 0.8548
    Epoch 5009/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3368 - accuracy: 0.8766
    Epoch 5010/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3358 - accuracy: 0.8675
    Epoch 5011/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3518 - accuracy: 0.8675
    Epoch 5012/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3451 - accuracy: 0.8711
    Epoch 5013/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3470 - accuracy: 0.8702
    Epoch 5014/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3467 - accuracy: 0.8711
    Epoch 5015/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3368 - accuracy: 0.8721
    Epoch 5016/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3348 - accuracy: 0.8684
    Epoch 5017/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3518 - accuracy: 0.8657
    Epoch 5018/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3434 - accuracy: 0.8702
    Epoch 5019/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3415 - accuracy: 0.8739
    Epoch 5020/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3328 - accuracy: 0.8766
    Epoch 5021/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3491 - accuracy: 0.8684
    Epoch 5022/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3486 - accuracy: 0.8648
    Epoch 5023/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3337 - accuracy: 0.8739
    Epoch 5024/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3341 - accuracy: 0.8739
    Epoch 5025/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3359 - accuracy: 0.8666
    Epoch 5026/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3396 - accuracy: 0.8684
    Epoch 5027/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3346 - accuracy: 0.8702
    Epoch 5028/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3343 - accuracy: 0.8702
    Epoch 5029/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3335 - accuracy: 0.8702
    Epoch 5030/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3348 - accuracy: 0.8721
    Epoch 5031/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3261 - accuracy: 0.8775
    Epoch 5032/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3401 - accuracy: 0.8675
    Epoch 5033/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3473 - accuracy: 0.8666
    Epoch 5034/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3403 - accuracy: 0.8684
    Epoch 5035/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3534 - accuracy: 0.8693
    Epoch 5036/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3373 - accuracy: 0.8730
    Epoch 5037/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3548 - accuracy: 0.8657
    Epoch 5038/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3445 - accuracy: 0.8775
    Epoch 5039/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3287 - accuracy: 0.8784
    Epoch 5040/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3421 - accuracy: 0.8730
    Epoch 5041/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3361 - accuracy: 0.8721
    Epoch 5042/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3508 - accuracy: 0.8693
    Epoch 5043/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3297 - accuracy: 0.8748
    Epoch 5044/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3384 - accuracy: 0.8711
    Epoch 5045/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3408 - accuracy: 0.8711
    Epoch 5046/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3419 - accuracy: 0.8684
    Epoch 5047/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3645 - accuracy: 0.8557
    Epoch 5048/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3298 - accuracy: 0.8739
    Epoch 5049/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3510 - accuracy: 0.8612
    Epoch 5050/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3361 - accuracy: 0.8784
    Epoch 5051/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3411 - accuracy: 0.8693
    Epoch 5052/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3307 - accuracy: 0.8766
    Epoch 5053/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3296 - accuracy: 0.8802
    Epoch 5054/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3306 - accuracy: 0.8721
    Epoch 5055/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3323 - accuracy: 0.8721
    Epoch 5056/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3476 - accuracy: 0.8684
    Epoch 5057/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3454 - accuracy: 0.8702
    Epoch 5058/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3280 - accuracy: 0.8757
    Epoch 5059/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3413 - accuracy: 0.8648
    Epoch 5060/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3360 - accuracy: 0.8702
    Epoch 5061/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3437 - accuracy: 0.8693
    Epoch 5062/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3568 - accuracy: 0.8539
    Epoch 5063/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3404 - accuracy: 0.8793
    Epoch 5064/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3400 - accuracy: 0.8793
    Epoch 5065/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3394 - accuracy: 0.8693
    Epoch 5066/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3373 - accuracy: 0.8748
    Epoch 5067/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3578 - accuracy: 0.8621
    Epoch 5068/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3467 - accuracy: 0.8684
    Epoch 5069/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3325 - accuracy: 0.8684
    Epoch 5070/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3384 - accuracy: 0.8675
    Epoch 5071/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3366 - accuracy: 0.8721
    Epoch 5072/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3464 - accuracy: 0.8612
    Epoch 5073/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3253 - accuracy: 0.8739
    Epoch 5074/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3354 - accuracy: 0.8766
    Epoch 5075/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3486 - accuracy: 0.8711
    Epoch 5076/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3357 - accuracy: 0.8657
    Epoch 5077/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3571 - accuracy: 0.8648
    Epoch 5078/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3460 - accuracy: 0.8657
    Epoch 5079/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3511 - accuracy: 0.8666
    Epoch 5080/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3450 - accuracy: 0.8675
    Epoch 5081/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3394 - accuracy: 0.8684
    Epoch 5082/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3342 - accuracy: 0.8748
    Epoch 5083/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3506 - accuracy: 0.8603
    Epoch 5084/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8666
    Epoch 5085/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3276 - accuracy: 0.8739
    Epoch 5086/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3355 - accuracy: 0.8757
    Epoch 5087/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3346 - accuracy: 0.8730
    Epoch 5088/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3418 - accuracy: 0.8702
    Epoch 5089/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3400 - accuracy: 0.8684
    Epoch 5090/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3425 - accuracy: 0.8684
    Epoch 5091/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3312 - accuracy: 0.8766
    Epoch 5092/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3696 - accuracy: 0.8566
    Epoch 5093/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3338 - accuracy: 0.8793
    Epoch 5094/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3264 - accuracy: 0.8748
    Epoch 5095/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3461 - accuracy: 0.8702
    Epoch 5096/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3422 - accuracy: 0.8784
    Epoch 5097/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3488 - accuracy: 0.8603
    Epoch 5098/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3322 - accuracy: 0.8702
    Epoch 5099/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3619 - accuracy: 0.8657
    Epoch 5100/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3368 - accuracy: 0.8739
    Epoch 5101/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3479 - accuracy: 0.8639
    Epoch 5102/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3280 - accuracy: 0.8739
    Epoch 5103/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3503 - accuracy: 0.8693
    Epoch 5104/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3653 - accuracy: 0.8539
    Epoch 5105/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3571 - accuracy: 0.8612
    Epoch 5106/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3398 - accuracy: 0.8711
    Epoch 5107/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3351 - accuracy: 0.8648
    Epoch 5108/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3364 - accuracy: 0.8711
    Epoch 5109/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3420 - accuracy: 0.8684
    Epoch 5110/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3436 - accuracy: 0.8657
    Epoch 5111/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3275 - accuracy: 0.8775
    Epoch 5112/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3392 - accuracy: 0.8739
    Epoch 5113/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3671 - accuracy: 0.8548
    Epoch 5114/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3259 - accuracy: 0.8775
    Epoch 5115/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3388 - accuracy: 0.8639
    Epoch 5116/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3329 - accuracy: 0.8775
    Epoch 5117/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3369 - accuracy: 0.8730
    Epoch 5118/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3369 - accuracy: 0.8657
    Epoch 5119/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3491 - accuracy: 0.8684
    Epoch 5120/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3349 - accuracy: 0.8784
    Epoch 5121/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3345 - accuracy: 0.8721
    Epoch 5122/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3291 - accuracy: 0.8811
    Epoch 5123/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3378 - accuracy: 0.8675
    Epoch 5124/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3609 - accuracy: 0.8639
    Epoch 5125/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3399 - accuracy: 0.8684
    Epoch 5126/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3483 - accuracy: 0.8648
    Epoch 5127/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3308 - accuracy: 0.8766
    Epoch 5128/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3274 - accuracy: 0.8702
    Epoch 5129/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3364 - accuracy: 0.8766
    Epoch 5130/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3508 - accuracy: 0.8684
    Epoch 5131/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3328 - accuracy: 0.8739
    Epoch 5132/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3534 - accuracy: 0.8666
    Epoch 5133/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3487 - accuracy: 0.8593
    Epoch 5134/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3348 - accuracy: 0.8721
    Epoch 5135/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3458 - accuracy: 0.8711
    Epoch 5136/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3357 - accuracy: 0.8693
    Epoch 5137/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3347 - accuracy: 0.8675
    Epoch 5138/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3292 - accuracy: 0.8766
    Epoch 5139/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3292 - accuracy: 0.8730
    Epoch 5140/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3314 - accuracy: 0.8775
    Epoch 5141/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3510 - accuracy: 0.8612
    Epoch 5142/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3269 - accuracy: 0.8802
    Epoch 5143/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3476 - accuracy: 0.8566
    Epoch 5144/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3390 - accuracy: 0.8684
    Epoch 5145/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8721
    Epoch 5146/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3440 - accuracy: 0.8775
    Epoch 5147/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3502 - accuracy: 0.8684
    Epoch 5148/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3283 - accuracy: 0.8811
    Epoch 5149/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3509 - accuracy: 0.8721
    Epoch 5150/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3279 - accuracy: 0.8721
    Epoch 5151/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3245 - accuracy: 0.8757
    Epoch 5152/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3355 - accuracy: 0.8675
    Epoch 5153/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3298 - accuracy: 0.8766
    Epoch 5154/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3285 - accuracy: 0.8766
    Epoch 5155/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3370 - accuracy: 0.8784
    Epoch 5156/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3315 - accuracy: 0.8766
    Epoch 5157/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3250 - accuracy: 0.8829
    Epoch 5158/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3346 - accuracy: 0.8657
    Epoch 5159/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3383 - accuracy: 0.8784
    Epoch 5160/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3325 - accuracy: 0.8730
    Epoch 5161/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3278 - accuracy: 0.8838
    Epoch 5162/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3292 - accuracy: 0.8766
    Epoch 5163/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3389 - accuracy: 0.8702
    Epoch 5164/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3424 - accuracy: 0.8721
    Epoch 5165/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3601 - accuracy: 0.8557
    Epoch 5166/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3383 - accuracy: 0.8657
    Epoch 5167/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3565 - accuracy: 0.8657
    Epoch 5168/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3321 - accuracy: 0.8684
    Epoch 5169/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3683 - accuracy: 0.8593
    Epoch 5170/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3334 - accuracy: 0.8775
    Epoch 5171/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3300 - accuracy: 0.8793
    Epoch 5172/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3325 - accuracy: 0.8757
    Epoch 5173/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3455 - accuracy: 0.8775
    Epoch 5174/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3332 - accuracy: 0.8829
    Epoch 5175/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3414 - accuracy: 0.8711
    Epoch 5176/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3429 - accuracy: 0.8775
    Epoch 5177/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3356 - accuracy: 0.8711
    Epoch 5178/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3464 - accuracy: 0.8666
    Epoch 5179/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3421 - accuracy: 0.8675
    Epoch 5180/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3380 - accuracy: 0.8711
    Epoch 5181/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3294 - accuracy: 0.8775
    Epoch 5182/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3474 - accuracy: 0.8666
    Epoch 5183/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3315 - accuracy: 0.8730
    Epoch 5184/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3376 - accuracy: 0.8666
    Epoch 5185/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3300 - accuracy: 0.8748
    Epoch 5186/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3357 - accuracy: 0.8702
    Epoch 5187/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3336 - accuracy: 0.8757
    Epoch 5188/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3323 - accuracy: 0.8766
    Epoch 5189/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3294 - accuracy: 0.8793
    Epoch 5190/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3353 - accuracy: 0.8748
    Epoch 5191/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3450 - accuracy: 0.8730
    Epoch 5192/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3337 - accuracy: 0.8793
    Epoch 5193/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3492 - accuracy: 0.8693
    Epoch 5194/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3534 - accuracy: 0.8711
    Epoch 5195/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3420 - accuracy: 0.8730
    Epoch 5196/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3301 - accuracy: 0.8693
    Epoch 5197/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3514 - accuracy: 0.8711
    Epoch 5198/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3354 - accuracy: 0.8711
    Epoch 5199/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3291 - accuracy: 0.8829
    Epoch 5200/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3461 - accuracy: 0.8739
    Epoch 5201/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3577 - accuracy: 0.8666
    Epoch 5202/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3302 - accuracy: 0.8702
    Epoch 5203/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3280 - accuracy: 0.8757
    Epoch 5204/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3526 - accuracy: 0.8666
    Epoch 5205/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3441 - accuracy: 0.8739
    Epoch 5206/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3275 - accuracy: 0.8684
    Epoch 5207/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3318 - accuracy: 0.8766
    Epoch 5208/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3296 - accuracy: 0.8775
    Epoch 5209/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3483 - accuracy: 0.8593
    Epoch 5210/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3329 - accuracy: 0.8793
    Epoch 5211/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3350 - accuracy: 0.8666
    Epoch 5212/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3422 - accuracy: 0.8666
    Epoch 5213/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3305 - accuracy: 0.8757
    Epoch 5214/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3336 - accuracy: 0.8766
    Epoch 5215/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3287 - accuracy: 0.8757
    Epoch 5216/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3440 - accuracy: 0.8693
    Epoch 5217/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3349 - accuracy: 0.8766
    Epoch 5218/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3346 - accuracy: 0.8711
    Epoch 5219/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3436 - accuracy: 0.8693
    Epoch 5220/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3338 - accuracy: 0.8757
    Epoch 5221/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3624 - accuracy: 0.8566
    Epoch 5222/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3713 - accuracy: 0.8557
    Epoch 5223/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3430 - accuracy: 0.8739
    Epoch 5224/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3495 - accuracy: 0.8684
    Epoch 5225/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3642 - accuracy: 0.8593
    Epoch 5226/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3326 - accuracy: 0.8721
    Epoch 5227/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3266 - accuracy: 0.8775
    Epoch 5228/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3318 - accuracy: 0.8811
    Epoch 5229/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3411 - accuracy: 0.8711
    Epoch 5230/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3274 - accuracy: 0.8739
    Epoch 5231/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3364 - accuracy: 0.8693
    Epoch 5232/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3518 - accuracy: 0.8593
    Epoch 5233/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3452 - accuracy: 0.8702
    Epoch 5234/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3292 - accuracy: 0.8775
    Epoch 5235/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3460 - accuracy: 0.8711
    Epoch 5236/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3443 - accuracy: 0.8639
    Epoch 5237/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3332 - accuracy: 0.8730
    Epoch 5238/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3372 - accuracy: 0.8711
    Epoch 5239/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3409 - accuracy: 0.8730
    Epoch 5240/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3433 - accuracy: 0.8739
    Epoch 5241/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3378 - accuracy: 0.8702
    Epoch 5242/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3228 - accuracy: 0.8775
    Epoch 5243/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3512 - accuracy: 0.8675
    Epoch 5244/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3491 - accuracy: 0.8684
    Epoch 5245/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3534 - accuracy: 0.8657
    Epoch 5246/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3471 - accuracy: 0.8739
    Epoch 5247/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3322 - accuracy: 0.8693
    Epoch 5248/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3490 - accuracy: 0.8603
    Epoch 5249/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3283 - accuracy: 0.8711
    Epoch 5250/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3298 - accuracy: 0.8730
    Epoch 5251/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3312 - accuracy: 0.8711
    Epoch 5252/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3633 - accuracy: 0.8639
    Epoch 5253/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3451 - accuracy: 0.8702
    Epoch 5254/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3351 - accuracy: 0.8721
    Epoch 5255/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3328 - accuracy: 0.8702
    Epoch 5256/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3395 - accuracy: 0.8657
    Epoch 5257/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3401 - accuracy: 0.8748
    Epoch 5258/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3321 - accuracy: 0.8684
    Epoch 5259/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3493 - accuracy: 0.8721
    Epoch 5260/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3541 - accuracy: 0.8711
    Epoch 5261/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3272 - accuracy: 0.8711
    Epoch 5262/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3352 - accuracy: 0.8739
    Epoch 5263/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3352 - accuracy: 0.8748
    Epoch 5264/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3287 - accuracy: 0.8775
    Epoch 5265/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3308 - accuracy: 0.8802
    Epoch 5266/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3305 - accuracy: 0.8793
    Epoch 5267/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3280 - accuracy: 0.8829
    Epoch 5268/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3341 - accuracy: 0.8730
    Epoch 5269/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3395 - accuracy: 0.8748
    Epoch 5270/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3375 - accuracy: 0.8784
    Epoch 5271/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3337 - accuracy: 0.8684
    Epoch 5272/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3408 - accuracy: 0.8766
    Epoch 5273/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3384 - accuracy: 0.8657
    Epoch 5274/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3332 - accuracy: 0.8639
    Epoch 5275/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3514 - accuracy: 0.8666
    Epoch 5276/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3291 - accuracy: 0.8757
    Epoch 5277/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3378 - accuracy: 0.8739
    Epoch 5278/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3720 - accuracy: 0.8530
    Epoch 5279/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3252 - accuracy: 0.8766
    Epoch 5280/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3393 - accuracy: 0.8711
    Epoch 5281/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3814 - accuracy: 0.8466
    Epoch 5282/6000
    35/35 [==============================] - 0s 676us/step - loss: 0.3375 - accuracy: 0.8766
    Epoch 5283/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3337 - accuracy: 0.8693
    Epoch 5284/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3489 - accuracy: 0.8648
    Epoch 5285/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3435 - accuracy: 0.8639
    Epoch 5286/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3454 - accuracy: 0.8621
    Epoch 5287/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3505 - accuracy: 0.8730
    Epoch 5288/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3300 - accuracy: 0.8757
    Epoch 5289/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3286 - accuracy: 0.8766
    Epoch 5290/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3340 - accuracy: 0.8739
    Epoch 5291/6000
    35/35 [==============================] - 0s 676us/step - loss: 0.3521 - accuracy: 0.8666
    Epoch 5292/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3431 - accuracy: 0.8721
    Epoch 5293/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3306 - accuracy: 0.8757
    Epoch 5294/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3318 - accuracy: 0.8757
    Epoch 5295/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3350 - accuracy: 0.8648
    Epoch 5296/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3470 - accuracy: 0.8730
    Epoch 5297/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3385 - accuracy: 0.8693
    Epoch 5298/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3361 - accuracy: 0.8657
    Epoch 5299/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3338 - accuracy: 0.8702
    Epoch 5300/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3297 - accuracy: 0.8784
    Epoch 5301/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3387 - accuracy: 0.8711
    Epoch 5302/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3368 - accuracy: 0.8711
    Epoch 5303/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3348 - accuracy: 0.8757
    Epoch 5304/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3326 - accuracy: 0.8739
    Epoch 5305/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3290 - accuracy: 0.8757
    Epoch 5306/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3241 - accuracy: 0.8793
    Epoch 5307/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3275 - accuracy: 0.8748
    Epoch 5308/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3293 - accuracy: 0.8775
    Epoch 5309/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3302 - accuracy: 0.8766
    Epoch 5310/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3392 - accuracy: 0.8612
    Epoch 5311/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3659 - accuracy: 0.8612
    Epoch 5312/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3459 - accuracy: 0.8612
    Epoch 5313/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3387 - accuracy: 0.8730
    Epoch 5314/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3327 - accuracy: 0.8748
    Epoch 5315/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3605 - accuracy: 0.8621
    Epoch 5316/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3270 - accuracy: 0.8811
    Epoch 5317/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3484 - accuracy: 0.8675
    Epoch 5318/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3441 - accuracy: 0.8730
    Epoch 5319/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3425 - accuracy: 0.8666
    Epoch 5320/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3502 - accuracy: 0.8630
    Epoch 5321/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3420 - accuracy: 0.8684
    Epoch 5322/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3374 - accuracy: 0.8648
    Epoch 5323/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3633 - accuracy: 0.8621
    Epoch 5324/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3411 - accuracy: 0.8693
    Epoch 5325/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3346 - accuracy: 0.8702
    Epoch 5326/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3340 - accuracy: 0.8829
    Epoch 5327/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3260 - accuracy: 0.8711
    Epoch 5328/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3252 - accuracy: 0.8748
    Epoch 5329/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3307 - accuracy: 0.8766
    Epoch 5330/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3276 - accuracy: 0.8793
    Epoch 5331/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3408 - accuracy: 0.8657
    Epoch 5332/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3385 - accuracy: 0.8693
    Epoch 5333/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3378 - accuracy: 0.8693
    Epoch 5334/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3338 - accuracy: 0.8711
    Epoch 5335/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3317 - accuracy: 0.8721
    Epoch 5336/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3382 - accuracy: 0.8739
    Epoch 5337/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3440 - accuracy: 0.8666
    Epoch 5338/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3289 - accuracy: 0.8784
    Epoch 5339/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3319 - accuracy: 0.8739
    Epoch 5340/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3358 - accuracy: 0.8711
    Epoch 5341/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3453 - accuracy: 0.8693
    Epoch 5342/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3400 - accuracy: 0.8711
    Epoch 5343/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3659 - accuracy: 0.8684
    Epoch 5344/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3391 - accuracy: 0.8757
    Epoch 5345/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3657 - accuracy: 0.8648
    Epoch 5346/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3397 - accuracy: 0.8702
    Epoch 5347/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3280 - accuracy: 0.8748
    Epoch 5348/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3291 - accuracy: 0.8757
    Epoch 5349/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3367 - accuracy: 0.8739
    Epoch 5350/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3336 - accuracy: 0.8711
    Epoch 5351/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3389 - accuracy: 0.8784
    Epoch 5352/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3343 - accuracy: 0.8748
    Epoch 5353/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3418 - accuracy: 0.8748
    Epoch 5354/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3577 - accuracy: 0.8639
    Epoch 5355/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8784
    Epoch 5356/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3329 - accuracy: 0.8621
    Epoch 5357/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3327 - accuracy: 0.8702
    Epoch 5358/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3329 - accuracy: 0.8702
    Epoch 5359/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3309 - accuracy: 0.8730
    Epoch 5360/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3350 - accuracy: 0.8684
    Epoch 5361/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3493 - accuracy: 0.8657
    Epoch 5362/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3351 - accuracy: 0.8721
    Epoch 5363/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3304 - accuracy: 0.8748
    Epoch 5364/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3288 - accuracy: 0.8857
    Epoch 5365/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3311 - accuracy: 0.8793
    Epoch 5366/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3387 - accuracy: 0.8748
    Epoch 5367/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3400 - accuracy: 0.8684
    Epoch 5368/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3407 - accuracy: 0.8775
    Epoch 5369/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3366 - accuracy: 0.8766
    Epoch 5370/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3315 - accuracy: 0.8784
    Epoch 5371/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3604 - accuracy: 0.8630
    Epoch 5372/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3311 - accuracy: 0.8739
    Epoch 5373/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3275 - accuracy: 0.8721
    Epoch 5374/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3435 - accuracy: 0.8739
    Epoch 5375/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3360 - accuracy: 0.8730
    Epoch 5376/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3323 - accuracy: 0.8739
    Epoch 5377/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3311 - accuracy: 0.8784
    Epoch 5378/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3516 - accuracy: 0.8693
    Epoch 5379/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3323 - accuracy: 0.8775
    Epoch 5380/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3310 - accuracy: 0.8684
    Epoch 5381/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3302 - accuracy: 0.8711
    Epoch 5382/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3336 - accuracy: 0.8775
    Epoch 5383/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3294 - accuracy: 0.8730
    Epoch 5384/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3412 - accuracy: 0.8766
    Epoch 5385/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3427 - accuracy: 0.8766
    Epoch 5386/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3277 - accuracy: 0.8784
    Epoch 5387/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3280 - accuracy: 0.8793
    Epoch 5388/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3327 - accuracy: 0.8702
    Epoch 5389/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3259 - accuracy: 0.8784
    Epoch 5390/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3282 - accuracy: 0.8757
    Epoch 5391/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3468 - accuracy: 0.8684
    Epoch 5392/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3590 - accuracy: 0.8675
    Epoch 5393/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3235 - accuracy: 0.8702
    Epoch 5394/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3376 - accuracy: 0.8666
    Epoch 5395/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3311 - accuracy: 0.8793
    Epoch 5396/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3448 - accuracy: 0.8730
    Epoch 5397/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3373 - accuracy: 0.8721
    Epoch 5398/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3399 - accuracy: 0.8711
    Epoch 5399/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3284 - accuracy: 0.8721
    Epoch 5400/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3487 - accuracy: 0.8657
    Epoch 5401/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3589 - accuracy: 0.8648
    Epoch 5402/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3539 - accuracy: 0.8630
    Epoch 5403/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3596 - accuracy: 0.8575
    Epoch 5404/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3506 - accuracy: 0.8675
    Epoch 5405/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3480 - accuracy: 0.8630
    Epoch 5406/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3276 - accuracy: 0.8784
    Epoch 5407/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3343 - accuracy: 0.8730
    Epoch 5408/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3387 - accuracy: 0.8702
    Epoch 5409/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3293 - accuracy: 0.8748
    Epoch 5410/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3296 - accuracy: 0.8811
    Epoch 5411/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3283 - accuracy: 0.8748
    Epoch 5412/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3471 - accuracy: 0.8702
    Epoch 5413/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3240 - accuracy: 0.8820
    Epoch 5414/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3305 - accuracy: 0.8702
    Epoch 5415/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3431 - accuracy: 0.8675
    Epoch 5416/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3388 - accuracy: 0.8766
    Epoch 5417/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3379 - accuracy: 0.8711
    Epoch 5418/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3331 - accuracy: 0.8775
    Epoch 5419/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3300 - accuracy: 0.8748
    Epoch 5420/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3693 - accuracy: 0.8575
    Epoch 5421/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3666 - accuracy: 0.8584
    Epoch 5422/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3429 - accuracy: 0.8657
    Epoch 5423/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3342 - accuracy: 0.8757
    Epoch 5424/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3343 - accuracy: 0.8693
    Epoch 5425/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3416 - accuracy: 0.8784
    Epoch 5426/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3259 - accuracy: 0.8784
    Epoch 5427/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3381 - accuracy: 0.8711
    Epoch 5428/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3502 - accuracy: 0.8775
    Epoch 5429/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3376 - accuracy: 0.8693
    Epoch 5430/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3295 - accuracy: 0.8675
    Epoch 5431/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3394 - accuracy: 0.8730
    Epoch 5432/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3363 - accuracy: 0.8730
    Epoch 5433/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3407 - accuracy: 0.8684
    Epoch 5434/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3430 - accuracy: 0.8630
    Epoch 5435/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3343 - accuracy: 0.8766
    Epoch 5436/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3458 - accuracy: 0.8675
    Epoch 5437/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3496 - accuracy: 0.8711
    Epoch 5438/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3278 - accuracy: 0.8775
    Epoch 5439/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3307 - accuracy: 0.8793
    Epoch 5440/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3299 - accuracy: 0.8775
    Epoch 5441/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3296 - accuracy: 0.8784
    Epoch 5442/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3457 - accuracy: 0.8739
    Epoch 5443/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3263 - accuracy: 0.8838
    Epoch 5444/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3303 - accuracy: 0.8775
    Epoch 5445/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3385 - accuracy: 0.8739
    Epoch 5446/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3339 - accuracy: 0.8766
    Epoch 5447/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3305 - accuracy: 0.8721
    Epoch 5448/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3472 - accuracy: 0.8621
    Epoch 5449/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3589 - accuracy: 0.8639
    Epoch 5450/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3459 - accuracy: 0.8657
    Epoch 5451/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3428 - accuracy: 0.8693
    Epoch 5452/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3268 - accuracy: 0.8811
    Epoch 5453/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8711
    Epoch 5454/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3296 - accuracy: 0.8757
    Epoch 5455/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3336 - accuracy: 0.8784
    Epoch 5456/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3423 - accuracy: 0.8630
    Epoch 5457/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3281 - accuracy: 0.8829
    Epoch 5458/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3501 - accuracy: 0.8612
    Epoch 5459/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3333 - accuracy: 0.8766
    Epoch 5460/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8730
    Epoch 5461/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3342 - accuracy: 0.8693
    Epoch 5462/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3260 - accuracy: 0.8802
    Epoch 5463/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3359 - accuracy: 0.8675
    Epoch 5464/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3292 - accuracy: 0.8739
    Epoch 5465/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3549 - accuracy: 0.8548
    Epoch 5466/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3349 - accuracy: 0.8766
    Epoch 5467/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3320 - accuracy: 0.8748
    Epoch 5468/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3314 - accuracy: 0.8711
    Epoch 5469/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3542 - accuracy: 0.8612
    Epoch 5470/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3333 - accuracy: 0.8739
    Epoch 5471/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3310 - accuracy: 0.8775
    Epoch 5472/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3530 - accuracy: 0.8584
    Epoch 5473/6000
    35/35 [==============================] - 0s 530us/step - loss: 0.3457 - accuracy: 0.8693
    Epoch 5474/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3367 - accuracy: 0.8711
    Epoch 5475/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3329 - accuracy: 0.8693
    Epoch 5476/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3529 - accuracy: 0.8621
    Epoch 5477/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3400 - accuracy: 0.8666
    Epoch 5478/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3359 - accuracy: 0.8675
    Epoch 5479/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3288 - accuracy: 0.8730
    Epoch 5480/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3293 - accuracy: 0.8775
    Epoch 5481/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3293 - accuracy: 0.8775
    Epoch 5482/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3519 - accuracy: 0.8693
    Epoch 5483/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3489 - accuracy: 0.8666
    Epoch 5484/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3336 - accuracy: 0.8721
    Epoch 5485/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3358 - accuracy: 0.8711
    Epoch 5486/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3866 - accuracy: 0.8485
    Epoch 5487/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3499 - accuracy: 0.8657
    Epoch 5488/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3486 - accuracy: 0.8630
    Epoch 5489/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3388 - accuracy: 0.8675
    Epoch 5490/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3372 - accuracy: 0.8711
    Epoch 5491/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3280 - accuracy: 0.8748
    Epoch 5492/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3302 - accuracy: 0.8757
    Epoch 5493/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3434 - accuracy: 0.8702
    Epoch 5494/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3261 - accuracy: 0.8748
    Epoch 5495/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3335 - accuracy: 0.8784
    Epoch 5496/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3302 - accuracy: 0.8757
    Epoch 5497/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3408 - accuracy: 0.8775
    Epoch 5498/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3573 - accuracy: 0.8648
    Epoch 5499/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3302 - accuracy: 0.8757
    Epoch 5500/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3363 - accuracy: 0.8739
    Epoch 5501/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3314 - accuracy: 0.8684
    Epoch 5502/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3684 - accuracy: 0.8521
    Epoch 5503/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3383 - accuracy: 0.8766
    Epoch 5504/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3537 - accuracy: 0.8702
    Epoch 5505/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3327 - accuracy: 0.8802
    Epoch 5506/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3366 - accuracy: 0.8711
    Epoch 5507/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3403 - accuracy: 0.8711
    Epoch 5508/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3409 - accuracy: 0.8666
    Epoch 5509/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3240 - accuracy: 0.8811
    Epoch 5510/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3427 - accuracy: 0.8693
    Epoch 5511/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3325 - accuracy: 0.8693
    Epoch 5512/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3303 - accuracy: 0.8793
    Epoch 5513/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8721
    Epoch 5514/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3323 - accuracy: 0.8748
    Epoch 5515/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3378 - accuracy: 0.8684
    Epoch 5516/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3436 - accuracy: 0.8730
    Epoch 5517/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3328 - accuracy: 0.8693
    Epoch 5518/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3435 - accuracy: 0.8675
    Epoch 5519/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3278 - accuracy: 0.8820
    Epoch 5520/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3418 - accuracy: 0.8739
    Epoch 5521/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3527 - accuracy: 0.8675
    Epoch 5522/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3437 - accuracy: 0.8630
    Epoch 5523/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3398 - accuracy: 0.8711
    Epoch 5524/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3470 - accuracy: 0.8639
    Epoch 5525/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3351 - accuracy: 0.8739
    Epoch 5526/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3338 - accuracy: 0.8748
    Epoch 5527/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3525 - accuracy: 0.8693
    Epoch 5528/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3294 - accuracy: 0.8684
    Epoch 5529/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3393 - accuracy: 0.8702
    Epoch 5530/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3361 - accuracy: 0.8693
    Epoch 5531/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3377 - accuracy: 0.8575
    Epoch 5532/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3505 - accuracy: 0.8684
    Epoch 5533/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3254 - accuracy: 0.8802
    Epoch 5534/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3385 - accuracy: 0.8675
    Epoch 5535/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3424 - accuracy: 0.8657
    Epoch 5536/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3271 - accuracy: 0.8775
    Epoch 5537/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3257 - accuracy: 0.8748
    Epoch 5538/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3314 - accuracy: 0.8702
    Epoch 5539/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3370 - accuracy: 0.8739
    Epoch 5540/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3451 - accuracy: 0.8684
    Epoch 5541/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3314 - accuracy: 0.8684
    Epoch 5542/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3306 - accuracy: 0.8784
    Epoch 5543/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3385 - accuracy: 0.8648
    Epoch 5544/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3302 - accuracy: 0.8757
    Epoch 5545/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3371 - accuracy: 0.8657
    Epoch 5546/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3378 - accuracy: 0.8775
    Epoch 5547/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3444 - accuracy: 0.8748
    Epoch 5548/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3368 - accuracy: 0.8684
    Epoch 5549/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3243 - accuracy: 0.8802
    Epoch 5550/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3273 - accuracy: 0.8757
    Epoch 5551/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3305 - accuracy: 0.8775
    Epoch 5552/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3887 - accuracy: 0.8530
    Epoch 5553/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3470 - accuracy: 0.8693
    Epoch 5554/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3246 - accuracy: 0.8757
    Epoch 5555/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3305 - accuracy: 0.8730
    Epoch 5556/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3391 - accuracy: 0.8739
    Epoch 5557/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3360 - accuracy: 0.8721
    Epoch 5558/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3303 - accuracy: 0.8766
    Epoch 5559/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3372 - accuracy: 0.8612
    Epoch 5560/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3390 - accuracy: 0.8702
    Epoch 5561/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3470 - accuracy: 0.8711
    Epoch 5562/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3471 - accuracy: 0.8721
    Epoch 5563/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3328 - accuracy: 0.8775
    Epoch 5564/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3343 - accuracy: 0.8739
    Epoch 5565/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3475 - accuracy: 0.8657
    Epoch 5566/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3295 - accuracy: 0.8766
    Epoch 5567/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3386 - accuracy: 0.8793
    Epoch 5568/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3331 - accuracy: 0.8711
    Epoch 5569/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3285 - accuracy: 0.8748
    Epoch 5570/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3430 - accuracy: 0.8684
    Epoch 5571/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3389 - accuracy: 0.8702
    Epoch 5572/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3592 - accuracy: 0.8612
    Epoch 5573/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3369 - accuracy: 0.8721
    Epoch 5574/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3411 - accuracy: 0.8612
    Epoch 5575/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3434 - accuracy: 0.8666
    Epoch 5576/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3454 - accuracy: 0.8702
    Epoch 5577/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3365 - accuracy: 0.8711
    Epoch 5578/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3720 - accuracy: 0.8593
    Epoch 5579/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3643 - accuracy: 0.8575
    Epoch 5580/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3429 - accuracy: 0.8639
    Epoch 5581/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3384 - accuracy: 0.8693
    Epoch 5582/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3427 - accuracy: 0.8603
    Epoch 5583/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3276 - accuracy: 0.8739
    Epoch 5584/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3341 - accuracy: 0.8711
    Epoch 5585/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3316 - accuracy: 0.8730
    Epoch 5586/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3430 - accuracy: 0.8766
    Epoch 5587/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3348 - accuracy: 0.8757
    Epoch 5588/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3270 - accuracy: 0.8838
    Epoch 5589/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3417 - accuracy: 0.8666
    Epoch 5590/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3284 - accuracy: 0.8793
    Epoch 5591/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3549 - accuracy: 0.8684
    Epoch 5592/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3297 - accuracy: 0.8748
    Epoch 5593/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3521 - accuracy: 0.8603
    Epoch 5594/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3396 - accuracy: 0.8820
    Epoch 5595/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3275 - accuracy: 0.8739
    Epoch 5596/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3324 - accuracy: 0.8757
    Epoch 5597/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3292 - accuracy: 0.8711
    Epoch 5598/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3356 - accuracy: 0.8702
    Epoch 5599/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3281 - accuracy: 0.8793
    Epoch 5600/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3370 - accuracy: 0.8711
    Epoch 5601/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3305 - accuracy: 0.8784
    Epoch 5602/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3292 - accuracy: 0.8775
    Epoch 5603/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3308 - accuracy: 0.8721
    Epoch 5604/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3319 - accuracy: 0.8648
    Epoch 5605/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3270 - accuracy: 0.8693
    Epoch 5606/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3502 - accuracy: 0.8693
    Epoch 5607/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3268 - accuracy: 0.8748
    Epoch 5608/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3392 - accuracy: 0.8748
    Epoch 5609/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3372 - accuracy: 0.8730
    Epoch 5610/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3376 - accuracy: 0.8711
    Epoch 5611/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3500 - accuracy: 0.8639
    Epoch 5612/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3403 - accuracy: 0.8666
    Epoch 5613/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3304 - accuracy: 0.8757
    Epoch 5614/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3440 - accuracy: 0.8684
    Epoch 5615/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3409 - accuracy: 0.8748
    Epoch 5616/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3488 - accuracy: 0.8612
    Epoch 5617/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3308 - accuracy: 0.8766
    Epoch 5618/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3271 - accuracy: 0.8775
    Epoch 5619/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3256 - accuracy: 0.8757
    Epoch 5620/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3295 - accuracy: 0.8730
    Epoch 5621/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3334 - accuracy: 0.8730
    Epoch 5622/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3310 - accuracy: 0.8739
    Epoch 5623/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3378 - accuracy: 0.8666
    Epoch 5624/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3451 - accuracy: 0.8702
    Epoch 5625/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3466 - accuracy: 0.8711
    Epoch 5626/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3548 - accuracy: 0.8584
    Epoch 5627/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3374 - accuracy: 0.8730
    Epoch 5628/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3360 - accuracy: 0.8684
    Epoch 5629/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3441 - accuracy: 0.8702
    Epoch 5630/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3284 - accuracy: 0.8739
    Epoch 5631/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3606 - accuracy: 0.8593
    Epoch 5632/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3648 - accuracy: 0.8621
    Epoch 5633/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3513 - accuracy: 0.8675
    Epoch 5634/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3266 - accuracy: 0.8711
    Epoch 5635/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3352 - accuracy: 0.8721
    Epoch 5636/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3253 - accuracy: 0.8766
    Epoch 5637/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3239 - accuracy: 0.8802
    Epoch 5638/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3446 - accuracy: 0.8666
    Epoch 5639/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3504 - accuracy: 0.8612
    Epoch 5640/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3296 - accuracy: 0.8730
    Epoch 5641/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3391 - accuracy: 0.8766
    Epoch 5642/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3541 - accuracy: 0.8702
    Epoch 5643/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3866 - accuracy: 0.8530
    Epoch 5644/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3408 - accuracy: 0.8711
    Epoch 5645/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3273 - accuracy: 0.8766
    Epoch 5646/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3268 - accuracy: 0.8784
    Epoch 5647/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3496 - accuracy: 0.8675
    Epoch 5648/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3294 - accuracy: 0.8711
    Epoch 5649/6000
    35/35 [==============================] - 0s 794us/step - loss: 0.3508 - accuracy: 0.8711
    Epoch 5650/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3279 - accuracy: 0.8802
    Epoch 5651/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3456 - accuracy: 0.8639
    Epoch 5652/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3412 - accuracy: 0.8793
    Epoch 5653/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3431 - accuracy: 0.8684
    Epoch 5654/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3306 - accuracy: 0.8766
    Epoch 5655/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3256 - accuracy: 0.8775
    Epoch 5656/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3286 - accuracy: 0.8757
    Epoch 5657/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3385 - accuracy: 0.8666
    Epoch 5658/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3332 - accuracy: 0.8721
    Epoch 5659/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3324 - accuracy: 0.8784
    Epoch 5660/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3429 - accuracy: 0.8702
    Epoch 5661/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3451 - accuracy: 0.8702
    Epoch 5662/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3341 - accuracy: 0.8775
    Epoch 5663/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3385 - accuracy: 0.8639
    Epoch 5664/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3317 - accuracy: 0.8748
    Epoch 5665/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3353 - accuracy: 0.8721
    Epoch 5666/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3596 - accuracy: 0.8548
    Epoch 5667/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3359 - accuracy: 0.8693
    Epoch 5668/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3282 - accuracy: 0.8702
    Epoch 5669/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3485 - accuracy: 0.8593
    Epoch 5670/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3306 - accuracy: 0.8748
    Epoch 5671/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3399 - accuracy: 0.8739
    Epoch 5672/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3331 - accuracy: 0.8748
    Epoch 5673/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3305 - accuracy: 0.8793
    Epoch 5674/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3230 - accuracy: 0.8775
    Epoch 5675/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3296 - accuracy: 0.8848
    Epoch 5676/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3340 - accuracy: 0.8702
    Epoch 5677/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3303 - accuracy: 0.8748
    Epoch 5678/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3358 - accuracy: 0.8748
    Epoch 5679/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3328 - accuracy: 0.8693
    Epoch 5680/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3293 - accuracy: 0.8748
    Epoch 5681/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3333 - accuracy: 0.8829
    Epoch 5682/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3357 - accuracy: 0.8757
    Epoch 5683/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3324 - accuracy: 0.8802
    Epoch 5684/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3281 - accuracy: 0.8793
    Epoch 5685/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3341 - accuracy: 0.8766
    Epoch 5686/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3321 - accuracy: 0.8711
    Epoch 5687/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3321 - accuracy: 0.8766
    Epoch 5688/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3282 - accuracy: 0.8757
    Epoch 5689/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3594 - accuracy: 0.8612
    Epoch 5690/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3269 - accuracy: 0.8775
    Epoch 5691/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3442 - accuracy: 0.8757
    Epoch 5692/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3254 - accuracy: 0.8784
    Epoch 5693/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3430 - accuracy: 0.8721
    Epoch 5694/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3497 - accuracy: 0.8612
    Epoch 5695/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3317 - accuracy: 0.8802
    Epoch 5696/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3636 - accuracy: 0.8593
    Epoch 5697/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3510 - accuracy: 0.8639
    Epoch 5698/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3440 - accuracy: 0.8630
    Epoch 5699/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3347 - accuracy: 0.8748
    Epoch 5700/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3402 - accuracy: 0.8748
    Epoch 5701/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3303 - accuracy: 0.8775
    Epoch 5702/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3309 - accuracy: 0.8829
    Epoch 5703/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3410 - accuracy: 0.8748
    Epoch 5704/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3299 - accuracy: 0.8757
    Epoch 5705/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3361 - accuracy: 0.8721
    Epoch 5706/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3637 - accuracy: 0.8657
    Epoch 5707/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3248 - accuracy: 0.8811
    Epoch 5708/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3270 - accuracy: 0.8784
    Epoch 5709/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3296 - accuracy: 0.8711
    Epoch 5710/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3382 - accuracy: 0.8711
    Epoch 5711/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3628 - accuracy: 0.8630
    Epoch 5712/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3287 - accuracy: 0.8775
    Epoch 5713/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3380 - accuracy: 0.8711
    Epoch 5714/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3550 - accuracy: 0.8666
    Epoch 5715/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3629 - accuracy: 0.8630
    Epoch 5716/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3604 - accuracy: 0.8621
    Epoch 5717/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3415 - accuracy: 0.8730
    Epoch 5718/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3297 - accuracy: 0.8739
    Epoch 5719/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3329 - accuracy: 0.8721
    Epoch 5720/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3395 - accuracy: 0.8711
    Epoch 5721/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3235 - accuracy: 0.8829
    Epoch 5722/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3515 - accuracy: 0.8666
    Epoch 5723/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3237 - accuracy: 0.8702
    Epoch 5724/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3482 - accuracy: 0.8612
    Epoch 5725/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3256 - accuracy: 0.8784
    Epoch 5726/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3330 - accuracy: 0.8811
    Epoch 5727/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3330 - accuracy: 0.8675
    Epoch 5728/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3467 - accuracy: 0.8648
    Epoch 5729/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3521 - accuracy: 0.8657
    Epoch 5730/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3409 - accuracy: 0.8675
    Epoch 5731/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3280 - accuracy: 0.8802
    Epoch 5732/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3290 - accuracy: 0.8666
    Epoch 5733/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3436 - accuracy: 0.8657
    Epoch 5734/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3355 - accuracy: 0.8702
    Epoch 5735/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3324 - accuracy: 0.8793
    Epoch 5736/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3354 - accuracy: 0.8711
    Epoch 5737/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3307 - accuracy: 0.8748
    Epoch 5738/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3492 - accuracy: 0.8639
    Epoch 5739/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3376 - accuracy: 0.8739
    Epoch 5740/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3337 - accuracy: 0.8693
    Epoch 5741/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3570 - accuracy: 0.8702
    Epoch 5742/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3349 - accuracy: 0.8675
    Epoch 5743/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3410 - accuracy: 0.8721
    Epoch 5744/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3258 - accuracy: 0.8748
    Epoch 5745/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3339 - accuracy: 0.8721
    Epoch 5746/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3415 - accuracy: 0.8711
    Epoch 5747/6000
    35/35 [==============================] - 0s 676us/step - loss: 0.3359 - accuracy: 0.8684
    Epoch 5748/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3292 - accuracy: 0.8802
    Epoch 5749/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3374 - accuracy: 0.8693
    Epoch 5750/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3391 - accuracy: 0.8702
    Epoch 5751/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3462 - accuracy: 0.8584
    Epoch 5752/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3382 - accuracy: 0.8721
    Epoch 5753/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3286 - accuracy: 0.8784
    Epoch 5754/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3312 - accuracy: 0.8657
    Epoch 5755/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3500 - accuracy: 0.8775
    Epoch 5756/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3327 - accuracy: 0.8739
    Epoch 5757/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3417 - accuracy: 0.8657
    Epoch 5758/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3415 - accuracy: 0.8693
    Epoch 5759/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3265 - accuracy: 0.8820
    Epoch 5760/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3266 - accuracy: 0.8748
    Epoch 5761/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3301 - accuracy: 0.8748
    Epoch 5762/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3279 - accuracy: 0.8829
    Epoch 5763/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3369 - accuracy: 0.8721
    Epoch 5764/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3371 - accuracy: 0.8657
    Epoch 5765/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3808 - accuracy: 0.8512
    Epoch 5766/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3915 - accuracy: 0.8521
    Epoch 5767/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3739 - accuracy: 0.8639
    Epoch 5768/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3757 - accuracy: 0.8584
    Epoch 5769/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3400 - accuracy: 0.8721
    Epoch 5770/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3415 - accuracy: 0.8748
    Epoch 5771/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3381 - accuracy: 0.8648
    Epoch 5772/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3500 - accuracy: 0.8675
    Epoch 5773/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3307 - accuracy: 0.8721
    Epoch 5774/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3293 - accuracy: 0.8757
    Epoch 5775/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3382 - accuracy: 0.8721
    Epoch 5776/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3412 - accuracy: 0.8721
    Epoch 5777/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3354 - accuracy: 0.8711
    Epoch 5778/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3377 - accuracy: 0.8757
    Epoch 5779/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3346 - accuracy: 0.8730
    Epoch 5780/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3450 - accuracy: 0.8711
    Epoch 5781/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3407 - accuracy: 0.8757
    Epoch 5782/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3441 - accuracy: 0.8684
    Epoch 5783/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3309 - accuracy: 0.8757
    Epoch 5784/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3422 - accuracy: 0.8666
    Epoch 5785/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3364 - accuracy: 0.8693
    Epoch 5786/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3364 - accuracy: 0.8757
    Epoch 5787/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3526 - accuracy: 0.8566
    Epoch 5788/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3262 - accuracy: 0.8766
    Epoch 5789/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3282 - accuracy: 0.8693
    Epoch 5790/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3275 - accuracy: 0.8784
    Epoch 5791/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3370 - accuracy: 0.8730
    Epoch 5792/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3263 - accuracy: 0.8775
    Epoch 5793/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3392 - accuracy: 0.8666
    Epoch 5794/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3259 - accuracy: 0.8766
    Epoch 5795/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3343 - accuracy: 0.8702
    Epoch 5796/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3366 - accuracy: 0.8675
    Epoch 5797/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3593 - accuracy: 0.8657
    Epoch 5798/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3336 - accuracy: 0.8775
    Epoch 5799/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3300 - accuracy: 0.8739
    Epoch 5800/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3469 - accuracy: 0.8730
    Epoch 5801/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3370 - accuracy: 0.8793
    Epoch 5802/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3394 - accuracy: 0.8711
    Epoch 5803/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3302 - accuracy: 0.8793
    Epoch 5804/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3537 - accuracy: 0.8639
    Epoch 5805/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3350 - accuracy: 0.8711
    Epoch 5806/6000
    35/35 [==============================] - 0s 882us/step - loss: 0.3272 - accuracy: 0.8820
    Epoch 5807/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3256 - accuracy: 0.8766
    Epoch 5808/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3421 - accuracy: 0.8757
    Epoch 5809/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3581 - accuracy: 0.8684
    Epoch 5810/6000
    35/35 [==============================] - 0s 676us/step - loss: 0.3314 - accuracy: 0.8721
    Epoch 5811/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3361 - accuracy: 0.8693
    Epoch 5812/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3292 - accuracy: 0.8757
    Epoch 5813/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3388 - accuracy: 0.8603
    Epoch 5814/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3334 - accuracy: 0.8739
    Epoch 5815/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3363 - accuracy: 0.8657
    Epoch 5816/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3632 - accuracy: 0.8675
    Epoch 5817/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3284 - accuracy: 0.8775
    Epoch 5818/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3261 - accuracy: 0.8766
    Epoch 5819/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3334 - accuracy: 0.8666
    Epoch 5820/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3347 - accuracy: 0.8802
    Epoch 5821/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3543 - accuracy: 0.8648
    Epoch 5822/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3498 - accuracy: 0.8593
    Epoch 5823/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3540 - accuracy: 0.8639
    Epoch 5824/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3364 - accuracy: 0.8730
    Epoch 5825/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3495 - accuracy: 0.8657
    Epoch 5826/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3347 - accuracy: 0.8748
    Epoch 5827/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3334 - accuracy: 0.8693
    Epoch 5828/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3505 - accuracy: 0.8621
    Epoch 5829/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3284 - accuracy: 0.8811
    Epoch 5830/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3359 - accuracy: 0.8757
    Epoch 5831/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3257 - accuracy: 0.8775
    Epoch 5832/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3350 - accuracy: 0.8693
    Epoch 5833/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3285 - accuracy: 0.8766
    Epoch 5834/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3261 - accuracy: 0.8775
    Epoch 5835/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3335 - accuracy: 0.8748
    Epoch 5836/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3352 - accuracy: 0.8693
    Epoch 5837/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3405 - accuracy: 0.8748
    Epoch 5838/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3415 - accuracy: 0.8739
    Epoch 5839/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3425 - accuracy: 0.8702
    Epoch 5840/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3343 - accuracy: 0.8766
    Epoch 5841/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3293 - accuracy: 0.8721
    Epoch 5842/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3365 - accuracy: 0.8702
    Epoch 5843/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3495 - accuracy: 0.8711
    Epoch 5844/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3282 - accuracy: 0.8802
    Epoch 5845/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3289 - accuracy: 0.8784
    Epoch 5846/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3288 - accuracy: 0.8748
    Epoch 5847/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3273 - accuracy: 0.8775
    Epoch 5848/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3307 - accuracy: 0.8811
    Epoch 5849/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3394 - accuracy: 0.8730
    Epoch 5850/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3276 - accuracy: 0.8766
    Epoch 5851/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3336 - accuracy: 0.8757
    Epoch 5852/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3474 - accuracy: 0.8630
    Epoch 5853/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3388 - accuracy: 0.8684
    Epoch 5854/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3439 - accuracy: 0.8693
    Epoch 5855/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3501 - accuracy: 0.8748
    Epoch 5856/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3506 - accuracy: 0.8639
    Epoch 5857/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3616 - accuracy: 0.8657
    Epoch 5858/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3401 - accuracy: 0.8648
    Epoch 5859/6000
    35/35 [==============================] - 0s 794us/step - loss: 0.3238 - accuracy: 0.8730
    Epoch 5860/6000
    35/35 [==============================] - 0s 853us/step - loss: 0.3322 - accuracy: 0.8784
    Epoch 5861/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3265 - accuracy: 0.8793
    Epoch 5862/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3422 - accuracy: 0.8603
    Epoch 5863/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3356 - accuracy: 0.8757
    Epoch 5864/6000
    35/35 [==============================] - 0s 677us/step - loss: 0.3405 - accuracy: 0.8693
    Epoch 5865/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3300 - accuracy: 0.8748
    Epoch 5866/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3360 - accuracy: 0.8666
    Epoch 5867/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3368 - accuracy: 0.8748
    Epoch 5868/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3439 - accuracy: 0.8657
    Epoch 5869/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3488 - accuracy: 0.8675
    Epoch 5870/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3314 - accuracy: 0.8793
    Epoch 5871/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3281 - accuracy: 0.8793
    Epoch 5872/6000
    35/35 [==============================] - 0s 794us/step - loss: 0.3239 - accuracy: 0.8711
    Epoch 5873/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3341 - accuracy: 0.8666
    Epoch 5874/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3250 - accuracy: 0.8757
    Epoch 5875/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3601 - accuracy: 0.8684
    Epoch 5876/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3252 - accuracy: 0.8757
    Epoch 5877/6000
    35/35 [==============================] - 0s 676us/step - loss: 0.3354 - accuracy: 0.8684
    Epoch 5878/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3369 - accuracy: 0.8721
    Epoch 5879/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3880 - accuracy: 0.8457
    Epoch 5880/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3366 - accuracy: 0.8757
    Epoch 5881/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3281 - accuracy: 0.8748
    Epoch 5882/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3261 - accuracy: 0.8730
    Epoch 5883/6000
    35/35 [==============================] - 0s 647us/step - loss: 0.3467 - accuracy: 0.8702
    Epoch 5884/6000
    35/35 [==============================] - 0s 676us/step - loss: 0.3351 - accuracy: 0.8757
    Epoch 5885/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3354 - accuracy: 0.8766
    Epoch 5886/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3268 - accuracy: 0.8775
    Epoch 5887/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3284 - accuracy: 0.8802
    Epoch 5888/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3586 - accuracy: 0.8666
    Epoch 5889/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3496 - accuracy: 0.8648
    Epoch 5890/6000
    35/35 [==============================] - 0s 618us/step - loss: 0.3448 - accuracy: 0.8711
    Epoch 5891/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3431 - accuracy: 0.8639
    Epoch 5892/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3356 - accuracy: 0.8757
    Epoch 5893/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3407 - accuracy: 0.8630
    Epoch 5894/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3331 - accuracy: 0.8748
    Epoch 5895/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3292 - accuracy: 0.8757
    Epoch 5896/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3368 - accuracy: 0.8693
    Epoch 5897/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3299 - accuracy: 0.8730
    Epoch 5898/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3279 - accuracy: 0.8838
    Epoch 5899/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3230 - accuracy: 0.8730
    Epoch 5900/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3469 - accuracy: 0.8666
    Epoch 5901/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3381 - accuracy: 0.8721
    Epoch 5902/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3340 - accuracy: 0.8748
    Epoch 5903/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3479 - accuracy: 0.8693
    Epoch 5904/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3268 - accuracy: 0.8766
    Epoch 5905/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3228 - accuracy: 0.8766
    Epoch 5906/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3342 - accuracy: 0.8711
    Epoch 5907/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3282 - accuracy: 0.8775
    Epoch 5908/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3273 - accuracy: 0.8766
    Epoch 5909/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3431 - accuracy: 0.8630
    Epoch 5910/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3332 - accuracy: 0.8730
    Epoch 5911/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3437 - accuracy: 0.8721
    Epoch 5912/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3557 - accuracy: 0.8666
    Epoch 5913/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3505 - accuracy: 0.8630
    Epoch 5914/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3616 - accuracy: 0.8603
    Epoch 5915/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3412 - accuracy: 0.8775
    Epoch 5916/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3384 - accuracy: 0.8757
    Epoch 5917/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3337 - accuracy: 0.8711
    Epoch 5918/6000
    35/35 [==============================] - 0s 470us/step - loss: 0.3462 - accuracy: 0.8630
    Epoch 5919/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3531 - accuracy: 0.8675
    Epoch 5920/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3266 - accuracy: 0.8811
    Epoch 5921/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3308 - accuracy: 0.8666
    Epoch 5922/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3329 - accuracy: 0.8721
    Epoch 5923/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3409 - accuracy: 0.8648
    Epoch 5924/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3382 - accuracy: 0.8721
    Epoch 5925/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3311 - accuracy: 0.8702
    Epoch 5926/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3428 - accuracy: 0.8702
    Epoch 5927/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3419 - accuracy: 0.8739
    Epoch 5928/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3410 - accuracy: 0.8739
    Epoch 5929/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3411 - accuracy: 0.8766
    Epoch 5930/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3320 - accuracy: 0.8820
    Epoch 5931/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3271 - accuracy: 0.8748
    Epoch 5932/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3247 - accuracy: 0.8775
    Epoch 5933/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3382 - accuracy: 0.8630
    Epoch 5934/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3298 - accuracy: 0.8757
    Epoch 5935/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3330 - accuracy: 0.8793
    Epoch 5936/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3875 - accuracy: 0.8548
    Epoch 5937/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3418 - accuracy: 0.8657
    Epoch 5938/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3362 - accuracy: 0.8757
    Epoch 5939/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3270 - accuracy: 0.8739
    Epoch 5940/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3445 - accuracy: 0.8684
    Epoch 5941/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3236 - accuracy: 0.8766
    Epoch 5942/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3270 - accuracy: 0.8802
    Epoch 5943/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3372 - accuracy: 0.8748
    Epoch 5944/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3384 - accuracy: 0.8757
    Epoch 5945/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3269 - accuracy: 0.8802
    Epoch 5946/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3385 - accuracy: 0.8693
    Epoch 5947/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3286 - accuracy: 0.8793
    Epoch 5948/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3433 - accuracy: 0.8721
    Epoch 5949/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3284 - accuracy: 0.8829
    Epoch 5950/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3304 - accuracy: 0.8748
    Epoch 5951/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3547 - accuracy: 0.8684
    Epoch 5952/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3394 - accuracy: 0.8802
    Epoch 5953/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3355 - accuracy: 0.8766
    Epoch 5954/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3451 - accuracy: 0.8639
    Epoch 5955/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3378 - accuracy: 0.8721
    Epoch 5956/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3274 - accuracy: 0.8748
    Epoch 5957/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3473 - accuracy: 0.8657
    Epoch 5958/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3367 - accuracy: 0.8721
    Epoch 5959/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3449 - accuracy: 0.8702
    Epoch 5960/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3363 - accuracy: 0.8721
    Epoch 5961/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3306 - accuracy: 0.8711
    Epoch 5962/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3407 - accuracy: 0.8739
    Epoch 5963/6000
    35/35 [==============================] - 0s 588us/step - loss: 0.3357 - accuracy: 0.8811
    Epoch 5964/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3330 - accuracy: 0.8693
    Epoch 5965/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3370 - accuracy: 0.8702
    Epoch 5966/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3624 - accuracy: 0.8557
    Epoch 5967/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3444 - accuracy: 0.8784
    Epoch 5968/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3346 - accuracy: 0.8693
    Epoch 5969/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3326 - accuracy: 0.8766
    Epoch 5970/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3251 - accuracy: 0.8730
    Epoch 5971/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3434 - accuracy: 0.8721
    Epoch 5972/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3563 - accuracy: 0.8548
    Epoch 5973/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3373 - accuracy: 0.8730
    Epoch 5974/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3702 - accuracy: 0.8612
    Epoch 5975/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.3313 - accuracy: 0.8775
    Epoch 5976/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3326 - accuracy: 0.8639
    Epoch 5977/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3292 - accuracy: 0.8675
    Epoch 5978/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3242 - accuracy: 0.8829
    Epoch 5979/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3333 - accuracy: 0.8793
    Epoch 5980/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3354 - accuracy: 0.8684
    Epoch 5981/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3436 - accuracy: 0.8693
    Epoch 5982/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3492 - accuracy: 0.8721
    Epoch 5983/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3675 - accuracy: 0.8593
    Epoch 5984/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3700 - accuracy: 0.8621
    Epoch 5985/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3256 - accuracy: 0.8802
    Epoch 5986/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3332 - accuracy: 0.8811
    Epoch 5987/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3310 - accuracy: 0.8811
    Epoch 5988/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3348 - accuracy: 0.8739
    Epoch 5989/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3252 - accuracy: 0.8730
    Epoch 5990/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3261 - accuracy: 0.8793
    Epoch 5991/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3343 - accuracy: 0.8757
    Epoch 5992/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3404 - accuracy: 0.8702
    Epoch 5993/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3520 - accuracy: 0.8630
    Epoch 5994/6000
    35/35 [==============================] - 0s 500us/step - loss: 0.3351 - accuracy: 0.8721
    Epoch 5995/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3407 - accuracy: 0.8739
    Epoch 5996/6000
    35/35 [==============================] - 0s 412us/step - loss: 0.3291 - accuracy: 0.8802
    Epoch 5997/6000
    35/35 [==============================] - 0s 471us/step - loss: 0.3379 - accuracy: 0.8739
    Epoch 5998/6000
    35/35 [==============================] - 0s 529us/step - loss: 0.4016 - accuracy: 0.8439
    Epoch 5999/6000
    35/35 [==============================] - 0s 441us/step - loss: 0.3456 - accuracy: 0.8684
    Epoch 6000/6000
    35/35 [==============================] - 0s 559us/step - loss: 0.3625 - accuracy: 0.8503





    <keras.callbacks.History at 0x197b83def40>




```python
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose= 1)
print('\nTest accuracy:', test_acc)
```

    12/12 [==============================] - 0s 455us/step - loss: 0.3113 - accuracy: 0.8804
    
    Test accuracy: 0.8804348111152649


Conclusion: After ruinning 6000 epochs, I was able to train a model to get 88.0% accuracy on my dataset. 

Final Conclusion: My dataset is linearly sparable as it has a high accuracy of 84.2 for both linear classifier and MLP. The accuracy goes even higher with more epochs using tensorflow. 

### Using tensorflow gave me the best accuracy so far for classifying my dataset. This shows why Artificial neural networks are so widely used to predictions in a variety of fields.
Through/images/out the duration of the coursework, I have learned different data exploration and pre processing techniques, bayers net, clustering, decision trees and neural networks.  



