#initial data exploration

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
#%%

#%%
df_train = pd.read_csv('data/train.csv')
df_train.head()
#%%

#%%
#target_0 = df_train.loc[df_train['target'] == 0]
#target_1 = df_train.loc[df_train['target'] == 1]
#this creates the EDA for each of the 300 variables in the list
#do not run with full 300 will crash the computer because of memory space ---- need to do it in batches of 100, and then pick ~ 10 that look interesting
# for i in range(0,10):
#     g = str(i)
#     # sns.distplot(target_0[[g]], hist=False, rug=True, color = 'black')
#     # sns.distplot(target_1[[g]], hist=False, rug=True, color = 'orange')
#     # plt.title("This is for variable %s" % g)
#     # plt.show()
#     print(g)
#%%


#%%
#using random trees classifier to identify feature importance
variablelist = []
#this iterates through the algorithm x times
for i in range(20):
    y = df_train['target']
    x = df_train.drop('target', axis=1)


# feature extraction
    model = ExtraTreesClassifier()
    model.fit(x, y)
    k = model.feature_importances_
    print(len(k))
    print(k)

#prints out the top 20 variables
    print(k.argsort()[-20:][::-1])
#appends them to a list for each iteration
    variablelist.extend(list(k.argsort()[-20:][::-1]))

    print(len(variablelist))
#counts how many times each variable appears
list(x.columns.values)
print(Counter(variablelist))
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(15)

k = Most_Common(variablelist)
j = [i[0] for i in k]
print(j)

target_0 = df_train.loc[df_train['target'] == 0]
target_1 = df_train.loc[df_train['target'] == 1]
#creates a graph for each variable showing their distinction
for i in j:
     g = str(i)
     sns.distplot(target_0[[g]], hist=False, rug=True, color = 'black')
     sns.distplot(target_1[[g]], hist=False, rug=True, color = 'orange')
     plt.title("This is for variable %s" % g)
     plt.show()
     # print(g)

#%%

### now we are able to do any nonlinear or linear transforamtion to the variables and see how it impacts the most important variables dynamically


