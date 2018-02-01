# %matplotlib inline
#
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.rc("font", size = 14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# importing the data set
data = pd.read_csv("bank-additional-full.csv", header = 0, sep = ";")
data.isnull().sum()
data = data.dropna()

#exploring  the data
data.describe(include="all")
data.shape
data['education'].unique()
data['job'].unique()
data['marital'].unique()
data['default'].unique()
data['housing'].unique()

# grouping similar kind of variables together
data['education'].unique()
data['education'] = np.where(data['education']=='basic.4y','Basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y','Basic', data['education'])
data['education'] = np.where(data['education']=='basic.9y','Basic', data['education'])
data['education'].unique()

# digging deep into the dataset
data['y'].value_counts()
sns.countplot(x='y',data=data,palette='hls')
plt.savefig('count_plot')

# looking into the distribution of yes vs no across all features
data.groupby('y').mean()
data['y'] = np.where(data['y']=="yes",1,data['y'])
data['y'] = np.where(data['y']=="no",0,data['y'])
data.groupby('job').mean()
data.groupby('education').mean()
data.groupby('marital').mean()

# adding some visualisations

pd.crosstab(data.job,data.y).plot(kind = "bar")
plt.title("Purchase frequency per job title")
plt.xlabel("job type")
plt.ylabel("Purchase decisiom")
plt.savefig("purchase freq vs job")
plt.figure().show()

table = pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked = True)
plt.title("stacked bar chart for marital status vs purchase")
plt.xlabel("marital status")
plt.ylabel("Proportion of customers")
plt.savefig('marital vs pur_stack')

table = pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked = True)
plt.title("stacked bar chart for Education vs purchase")
plt.xlabel("Education")
plt.ylabel("Proportion of customers")
plt.savefig('Education vs pur_stack')

pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('purchase frequency for day of week')
plt.xlabel("day of week")
plt.ylabel('frequency of purchase')
plt.savefig('pur_dayofweek_bar')

pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('purchase frequency for Month')
plt.xlabel("month")
plt.ylabel('frequency of purchase monthly')
plt.savefig('purchase vs month')

data.age.hist()
plt.title('Histogram of age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')

pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('purchase frequency for poutcome')
plt.xlabel("poutcome")
plt.ylabel("frequency of purchase")
plt.savefig("purchase_fre_pout_bar")

# creating dummy variables
cat_var=['job','marital','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_var:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(data[var],prefix=var,drop_first=True)
    data1 = data.join(cat_list)
    data = data1

cat_vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

data_final = data[to_keep]

data_final_vars = data_final.columns.values.tolist()
y = ['y']
X = [i for i in data_final_vars if i not in y]
data_X = data_final[X]

# feature selection
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg,55)
rfe = rfe.fit(data_final[X],np.asanyarray(data_final[y][:],dtype=float))

print rfe.support_
print rfe.ranking_

X = data_final[X]
y = data_final[y]

# model implementation
import statsmodels.api as sm
logit_model = sm.Logit(np.asanyarray(y,dtype=float),np.asanyarray(X,dtype=float))
result = logit_model.fit()
print result.summary()

# logistic regression model fitting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train,np.asanyarray(y_train,dtype=float))

# predicting the test dataset
y_pred = logreg.predict(X_test)
format(logreg.score(X_test,np.asanyarray(y_test,dtype=float)))

# cross- validating the model to check if there is overfitting
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10,random_state=3)
model_cv = LogisticRegression()
results = model_selection.cross_val_score(model_cv,np.asanyarray(X_train),np.asanyarray(y_train,dtype=float),cv=kfold,scoring='accuracy')
print "accuracy with 10 fold cross validation : %.3f" %(results.mean())

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.asanyarray(y_test,dtype=float),y_pred)
from sklearn.metrics import classification_report
print classification_report(np.asanyarray(y_test,dtype=float),y_pred)


# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(np.asanyarray(y_test,dtype=float), logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(np.asanyarray(y_test,dtype=float), logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
plt.savefig("roc curve")


