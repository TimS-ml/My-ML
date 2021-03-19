# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# #!pip3 install lightgbm

# %% [markdown]
# ### Intuition of Quantile Loss 

# %% [markdown]
# Typical Loss Function:
#
# $$ L = (y - X\theta)^2 $$
#
# Quantile Loss:
#
# $$
# \begin{equation}
#   L =
#     \begin{cases}
#       \tau (y - X\theta) & \text{if $y - X\theta\ge 0$}\\
#       (\tau - 1) (y - X\theta) & \text{if $y - X\theta < 0$ }\\
#     \end{cases}       
# \end{equation}
# $$
#
# We want to penilze loss if:
# - the percentile $\tau$ is low, but the prediction $X\theta$ is high
# - the percentile $\tau$ is high, but the prediction $X\theta$ is low

# %% [markdown]
# ### Problem

# %% [markdown]
# Let's build a regression model that determines delivery time based on distance of house from store.

# %% [markdown]
# ### Build Dataset

# %%
from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

# %%
X, y = make_regression(n_samples=10000,
                n_features=1,
                n_informative=1,
                n_targets=1,
                random_state=42)

Xs = pd.DataFrame(X, columns = ['distance'])
ys = pd.DataFrame(y, columns = ['time_to_buyer'])

Xs['distance'] = Xs['distance'].apply(lambda x: 10 + 2 * (x + np.random.normal(loc=1))  )
ys['time_to_buyer'] = ys['time_to_buyer'].apply(lambda x: 60 + 0.3* (x + np.random.normal(loc=1)) )

# %%
df = Xs.merge(ys, left_index=True, right_index=True)

# %%
df.describe()

# %%
train_df, test_df = train_test_split(df, test_size=0.10, shuffle=False)

# %%
X_train, y_train = train_df[['distance']], train_df[['time_to_buyer']]
X_test, y_test = test_df[['distance']], test_df[['time_to_buyer']]

# %% [markdown]
# ### Visualize Data 

# %%
plt.rcParams.update({'font.size': 10, 'figure.dpi':100})
sns.scatterplot(data=test_df, x="distance", y="time_to_buyer", marker='+')
plt.grid(linestyle='-', linewidth=0.2)

# %%
classifiers = {}
for tau in [0.1, 0.5, 0.9]:
    clf = LGBMRegressor(objective='quantile', alpha=tau)
    clf.fit(X_train, y_train)
    preds = pd.DataFrame(clf.predict(X_test), columns = [str(tau)])
    classifiers[str(tau)] = {'clf': clf, 'predictions': preds}

# %%
data = pd.DataFrame({'distance': X_test.reset_index()['distance'],
              '0.1': classifiers['0.1']['predictions']['0.1'],
              '0.5': classifiers['0.5']['predictions']['0.5'],
              '0.9': classifiers['0.9']['predictions']['0.9'],
              'time_to_buyer': y_test.reset_index()['time_to_buyer']})

# %%
data.sample(2)

# %%
melted_data = pd.melt(data, id_vars=['distance'])

# %%
melted_data

# %%
plt.rcParams.update({'font.size': 10, 'figure.dpi':100})
sns.scatterplot(data=melted_data, 
                x="distance",
                y='value',
                hue='variable',
                hue_order=['time_to_buyer', '0.1', '0.5', '0.9'],
                marker='+')
plt.grid(linestyle='-', linewidth=0.2)
ylabel = plt.ylabel("Time (Minutes)")
xlabel = plt.xlabel("Distance (Miles)")

# %%
(data['time_to_buyer'] > data['0.1']).value_counts()

# %%
(data['time_to_buyer'] > data['0.5']).value_counts()

# %%
(data['time_to_buyer'] > data['0.9']).value_counts()

# %% [markdown]
# ### Notes
#
# - Instacart: https://tech.instacart.com/how-instacart-delivers-on-time-using-quantile-regression-2383e2e03edb
# - Quantile Regression blog: http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.html
