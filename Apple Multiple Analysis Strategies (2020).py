#!/usr/bin/env python
# coding: utf-8

#    APPLE $AAPL 2020 Analysis ---- Taylor Bommarito
#    
# 1. Probability and VaR Calculation
# 2. Z-score and P-value Calculation
# 3. Association Between Random Variables in AAPL

# In[1]:


#Libraries
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from scipy.stats import norm
from scipy.special import ndtr as ndtr
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
np.random.seed(8)


# =====================================================================================

# In[2]:


#AAPL Data
aapl = web.DataReader('AAPL', data_source='yahoo', start='2019-11-01')
aapl['MA20'] = aapl['Close'].rolling(20).mean()
aapl['MA50'] = aapl['Close'].rolling(50).mean()
aapl = aapl.dropna()
aapl


# =====================================================================================

# Probability and VaR Calculation

# In[3]:


#Calculation
aapl['logReturn'] = np.log(aapl['Close'].shift(-1)) - np.log(aapl['Close'])


# In[4]:


#Calculations and Histogram
mu = aapl['logReturn'].mean()
sigma = aapl['logReturn'].std(ddof=1)

density = pd.DataFrame()
density['x'] = np.arange(aapl['logReturn'].min()-0.01, aapl['logReturn'].max()+0.01, 0.001)
density['pdf'] = norm.pdf(density['x'], mu, sigma)

aapl['logReturn'].hist(bins=50, figsize=(18, 10))
plt.plot(density['x'], density['pdf'], color='red')
plt.show()


# In[5]:


#Probability
prob_return1 = norm.cdf(-0.05, mu, sigma)
print('The probability is ', prob_return1)


# In[6]:


#Probability
mu220 = 220*mu
sigma220= (220**0.5) * sigma
print('The probability of dropping over 40% in 220 days is ', norm.cdf(-0.4, mu220, sigma220))


# In[7]:


#Probability
mu220 = 220*mu
sigma220 = (220**0.5) * sigma
drop20 = norm.cdf(-0.2, mu220, sigma220)
print('The probability of dropping over 20% in 220 days is ', drop20)


# In[8]:


#Value at Risk
VaR = norm.ppf(0.05, mu, sigma)
print('Single day Value at Risk ', VaR)


# In[9]:


#Value at Risk
print('5% quantile ', norm.ppf(0.05, mu, sigma))
print('95%quantile', norm.ppf(0.95, mu, sigma))


# In[10]:


#Value at Risk
q25 = norm.ppf(0.25, mu, sigma)
print('25% quantile', q25)
q75 = norm.ppf(0.75, mu, sigma)
print('75% quantile', q75)


# =====================================================================================

# Z-score and P-value Calculation

# In[11]:


#size, mu, sigma calc
sample_size = aapl['logReturn'].shape[0]
print(sample_size)
print(mu)
print(sigma)


# In[12]:


#Zscore Calc
norm_dist = np.random.normal(mu, sigma, 152)


# In[13]:


#Zscore Calc
norm_dist


# In[14]:


#Zscore Calc
df = pd.DataFrame(norm_dist,columns=['Data'])
df.head()


# In[15]:


#Zscore Calc
df.tail()


# In[16]:


#Zscore Calc
for col in df.columns:
    col_zscore = col + '_zscore'
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)


# In[17]:


#Zscore Calc
df.tail()


# In[18]:


#Pvalue Calc
df['Data_p_values'] = 1 - ndtr(df['Data_zscore'])


# In[19]:


#Pvalue Calc
df.tail()


# In[20]:


#Two Tailed Test
alpha = 0.025


# In[21]:


#Calc
df['statistically_significant'] = (df.Data_p_values > alpha).astype(int)


# In[22]:


#Calc
df.statistically_significant.value_counts()


# In[23]:


#Result
df.loc[df.statistically_significant == 0, :]


# In[24]:


#Result Graphs
df.hist(bins=50, figsize=(18, 10))


# =====================================================================================

# Association Between Random Variables in AAPL

# In[25]:


#Multiple Graph Possibilities
from pandas.plotting import scatter_matrix
sm = scatter_matrix(aapl, figsize=(24, 24))


# In[26]:


#Volume and Close Price Graph
aapl.plot(kind='scatter', x='Close', y='Volume', figsize=(10, 10))
plt.title('Close / Volume Scatter')


# =====================================================================================
