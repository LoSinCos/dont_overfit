#%% Load libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

#%%library adjustments
pd.set_option('display.float_format', lambda x: '%.3f' % x)
sns.set(style='white', context='notebook', palette='deep')
#warnings.filterwarnings('ignore')
sns.set_style('white')
#%matplotlib inline
# Set some options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)

#%% Read in data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#%% preview the data
#shape
print(train.shape)
print(test.shape)

#summary
print(train.info())

#head
print(train.head())

# View head
train.head(10)

# show first 30 lines and bottom 30 lines
display(train)

#%% create histogram of training data targets
sns.countplot(x="target", data=train, palette="Greens_d")
plt.show()

#%%
print(train['0'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(train['0'], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.show()

#%%
sns.distplot(train.0[])