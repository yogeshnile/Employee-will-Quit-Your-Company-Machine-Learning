# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import cufflinks as cf
import sklearn

# %%
pyo.init_notebook_mode(connected=True)
cf.go_offline()

# %%
"""
### Import Data
"""

# %%
hr = pd.read_csv('hr_data.csv')

# %%
hr.shape

# %%
hr.size

# %%
hr.info()

# %%
hr['department'].unique()

# %%
hr['salary'].unique()

# %%
s_hr = pd.read_excel('employee_satisfaction_evaluation.xlsx')

# %%
s_hr.info()

# %%
"""
### Join Both Data
"""

# %%
hr = hr.set_index('employee_id').join(s_hr.set_index('EMPLOYEE #'))

# %%
hr = hr.reset_index()

# %%
hr

# %%
hr.info()

# %%
hr[hr.isnull().any(axis=1)]

# %%
hr.describe()

# %%
"""
### Fill Null Value
"""

# %%
hr.fillna(hr.mean(), inplace=True)

# %%
hr[hr.isnull().any(axis=1)]

# %%
hr.drop('employee_id', axis=1, inplace=True)

# %%
hr.info()

# %%
hr.groupby('department').sum()

# %%
hr.groupby('department').mean()

# %%
hr['department'].value_counts()

# %%
hr['left'].value_counts()

# %%
employee_condition = ['Employee will stay','employee will leave']

# %%
"""
### Check Corr between Data
"""

# %%
def plot_corr(df,size=10):
    corr = df.corr()
    fig,ax=plt.subplots(figsize=(size,size))
    ax.legend()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)

# %%
plot_corr(hr)

# %%
hr.corr()

# %%
"""
### Data Visulization
"""

# %%
sns.barplot(x='left', y='satisfaction_level', data=hr)

# %%
sns.barplot(x='promotion_last_5years', y='satisfaction_level', data=hr)

# %%
sns.barplot(x='time_spend_company', y='satisfaction_level', data=hr, hue='left')

# %%
sns.pairplot(hr, hue='left')

# %%
y = hr[['department','salary']]

# %%
"""
### Convert Data String into number
"""

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_le = le.fit_transform(hr['salary'])

# %%
y_le

# %%
hr['salary_num'] = y_le

# %%
Salary = ['high','low','medium']

# %%
hr.drop('salary', axis=1, inplace=True)

# %%
dep = LabelEncoder()
dep_num = dep.fit_transform(hr['department'])

# %%
dep_num

# %%
hr['department_num'] = dep_num

# %%
hr['department'].unique()

# %%
Department = ['IT','RandD','accounting','hr','management','marketing','product_mng','sales','support','technical']

# %%
hr.loc[hr['department_num']== 5].head(1)

# %%
hr.drop('department', axis=1, inplace=True)

# %%
x = hr.drop('left', axis=1)

# %%
y = hr['left']

# %%
"""
### Train test split
"""

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# %%
"""
### Decision Tree
"""

# %%
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

# %%
pred_dt = dt.predict(x_test)

# %%
from sklearn.metrics import accuracy_score

# %%
accuracy_score(y_test, pred_dt)*100

# %%
"""
### KNN
"""

# %%
from sklearn.preprocessing import StandardScaler

# %%
sc = StandardScaler().fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_std, y_train)

# %%
pred_knn = knn.predict(x_test_std)

# %%
accuracy_score(y_test, pred_knn)*100

# %%
"""
### Choose Best n_neighbors value
"""

# %%
scores = {}
for i in range(1,26):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_std, y_train)
    pred_knn = knn.predict(x_test_std)
    scores[i] = accuracy_score(y_test, pred_knn)*100

# %%
scores