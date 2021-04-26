import io
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# sns.set_context("paper", rc={"font.size":20,"axes.titlesize":22,"axes.labelsize":16})
sns.set_style("darkgrid")
sns.set_context("talk")
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


url = 'https://raw.githubusercontent.com/TimS-ml/DataMining/master/0_TakeHome/0x01_conversion_project.csv'
f = requests.get(url).content
data = pd.read_csv(io.StringIO(f.decode('utf-8')))

# data = pd.read_csv('./conversion_project.csv')
data.head()


data.describe()


# data.country.mode()


for column in data.columns:
    print(data[column].value_counts().nlargest(5))
    print()


conversion_rate = 10200 / (306000 + 10200)
print('Conversion rate: {:2.2get_ipython().run_line_magic("}'.format(conversion_rate))", "")


data[data['age'] > 79].head()


data = data[data['age'] <= 79]


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

sns.countplot(x='country', hue='converted', data=data, ax=ax[0])
ax[0].set_title('Count Plot of Country')
ax[0].set_yscale('log')

sns.barplot(x='country', y='converted', data=data, ax=ax[1]);
ax[1].set_title('Mean Conversion Rate per Country')
plt.tight_layout()

plt.show()


grouped = data[['country', 'converted']].groupby('country').mean().reset_index()
grouped.head()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

sns.countplot(x='new_user', hue='converted', data=data, ax=ax[0])
ax[0].set_title('Count Plot of User Types')
ax[0].set_yscale('log')

sns.barplot(x='new_user', y='converted', data=data, ax=ax[1]);
ax[1].set_title('Mean Conversion Rate per User Type')
plt.tight_layout()

plt.show()


grouped = data[['new_user', 'converted']].groupby('new_user').mean().reset_index()
grouped.head()


grouped = data[['source', 'converted']].groupby('source').mean().reset_index()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='source', hue='converted', data=data, ax=ax[0])
ax[0].set_title('Count Plot of Different Sources', fontsize=16)
ax[0].set_yscale('log')
sns.barplot(x='source', y='converted', data=data, ax=ax[1]);
ax[1].set_title('Mean Conversion Rate per Source', fontsize=16)
plt.tight_layout()
plt.show()


grouped = data[['age', 'converted']].groupby('age').mean().reset_index()
grouped.head()


hist_kws = {'histtype': 'bar', 'edgecolor': 'black', 'alpha': 0.2}

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

sns.distplot(data[data['converted'] == 0]['age'], label='Converted 0', 
             ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['converted'] == 1]['age'], label='Converted 1', 
             ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Count Plot of Age')
ax[0].legend()

ax[1].plot(grouped['age'], grouped['converted'], '.-')
ax[1].set_title('Mean Conversion Rate vs. Age')
ax[1].set_xlabel('age')
ax[1].set_ylabel('Mean convertion rate')
ax[1].grid(True)

plt.show()


grouped = data[['total_pages_visited', 'converted']].groupby('total_pages_visited').mean().reset_index()
grouped.head()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

sns.distplot(data[data['converted'] == 0]['total_pages_visited'], 
             label='Converted 0', ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['converted'] == 1]['total_pages_visited'], 
             label='Converted 1', ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Count Plot of Age')
ax[0].legend()

ax[1].plot(grouped['total_pages_visited'], grouped['converted'], '.-')
ax[1].set_title('Mean Conversion Rate vs. Total_pages_visited')
ax[1].set_xlabel('total_pages_visited')
ax[1].set_ylabel('Mean convertion rate')
ax[1].grid(True)

plt.show()



