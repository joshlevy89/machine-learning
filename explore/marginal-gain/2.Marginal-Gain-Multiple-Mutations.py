
# coding: utf-8

# # Compare logistic regression models of several mutations for a) covariates only and b) covariates with gene expression data to determine marginal gain using gene expression data 

# In[4]:

import os
import urllib
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.feature_selection import SelectKBest
from statsmodels.robust.scale import mad
from IPython.display import display
import gc


# In[5]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Load Data

# In[6]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', '..', 'download', 'covariates.tsv')\ncovariates = pd.read_table(path, index_col=0)")


# In[7]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', '..', 'download', 'expression-matrix.tsv.bz2')\nexpression = pd.read_table(path, index_col=0)")


# In[40]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..','..','download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[41]:

X = {}
X['a'] = covariates
X['b'] = pd.concat([covariates,expression], axis=1)
for k in ['a','b']:
    print(X[k].shape)


# In[42]:

mutations = {
    '7157': 'TP53',   # tumor protein p53
    '7428': 'VHL',    # von Hippel-Lindau tumor suppressor
    '29126': 'CD274', # CD274 molecule
    '672': 'BRCA1',   # BRCA1, DNA repair associated
    '675': 'BRCA2',   # BRCA2, DNA repair associated
    '238': 'ALK',     # anaplastic lymphoma receptor tyrosine kinase
    '4221': 'MEN1',   # menin 1
    '5979': 'RET',    # ret proto-oncogene
}


# ## Median absolute deviation feature selection

# In[43]:

def fs_mad(x, y):
    """    
    Get the median absolute deviation (MAD) for each column of x
    """
    scores = mad(x) 
    return scores, np.array([np.NaN]*len(scores))


# ## Define pipeline and Cross validation model fitting

# In[44]:

# Parameter Sweep for Hyperparameters
param_grid = {
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0],
}

pipeline = Pipeline(steps=[
    ('imputer', Imputer()),    
    ('select', SelectKBest(fs_mad)),
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])

cv_pipeline = {}
for k in ['a','b']:
    pg = param_grid.copy()
    if k == 'a': pg['select__k'] = ['all']
    elif k=='b': pg['select__k'] = [2000]
    cv_pipeline[k] =  GridSearchCV(estimator=pipeline, param_grid=pg, scoring='roc_auc')


# In[45]:

def get_aurocs(X, y, pipeline, series):
    """
    Fit the classifier for the given mutation (y) and output predictions for it
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    pipeline.fit(X=X_train, y=y_train)
    y_pred_train = pipeline.decision_function(X_train)
    y_pred_test = pipeline.decision_function(X_test)
    cv_score_df = grid_scores_to_df(pipeline.grid_scores_)
    series['mean_cv_auroc'] = cv_score_df.score.max()
    series['training_auroc'] = roc_auc_score(y_train, y_pred_train)
    series['testing_auroc'] = roc_auc_score(y_test, y_pred_test)
    return series

def grid_scores_to_df(grid_scores):
    """
    Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to 
    a tidy pandas DataFrame where each row is a hyperparameter-fold combinatination.
    """
    rows = list()
    for grid_score in grid_scores:
        for fold, score in enumerate(grid_score.cv_validation_scores):
            row = grid_score.parameters.copy()
            row['fold'] = fold
            row['score'] = score
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

auroc_df = {}


# In[46]:

get_ipython().run_cell_magic('time', '', '# Train model a: covariates only.\nwarnings.filterwarnings("ignore") # ignore deprecation warning for grid_scores_\nrows = list()\nfor m in list(mutations):\n    series = pd.Series()\n    series[\'mutation\'] = m\n    series[\'symbol\'] = mutations[m]\n    rows.append(get_aurocs(X[\'a\'], Y[m], cv_pipeline[\'a\'], series))\nauroc_df[\'a\'] = pd.DataFrame(rows)\nauroc_df[\'a\'].sort_values([\'symbol\', \'testing_auroc\'], ascending=[True, False], inplace=True)')


# In[47]:

get_ipython().run_cell_magic('time', '', '# Train model b: covariates with gene expression data.\nwarnings.filterwarnings("ignore") # ignore deprecation warning for grid_scores_\nrows = list()\nfor m in list(mutations):\n    series = pd.Series()\n    series[\'mutation\'] = m\n    series[\'symbol\'] = mutations[m]\n    rows.append(get_aurocs(X[\'b\'], Y[m], cv_pipeline[\'b\'], series))\nauroc_df[\'b\'] = pd.DataFrame(rows)\nauroc_df[\'b\'].sort_values([\'symbol\', \'testing_auroc\'], ascending=[True, False], inplace=True)')


# In[50]:

display(auroc_df['a'])
display(auroc_df['b'])


# In[65]:

auroc_df['a'].to_csv('auroc_covariates_only.tsv', index=False, sep='\t', float_format='%.5g')
auroc_df['b'].to_csv('auroc_covariates_and_expression.tsv', index=False, sep='\t', float_format='%.5g')
auroc_df['c'] = auroc_df['b'].loc[:,'mean_cv_auroc':]-auroc_df['a'].loc[:,'mean_cv_auroc':]
auroc_df['c'][['mutation', 'symbol']] = auroc_df['b'].loc[:, ['mutation', 'symbol']]
auroc_df['c']


# # Covariates only vs covariates+expression model

# In[68]:

plot_df = pd.melt(auroc_df['c'], id_vars='symbol', value_vars=['mean_cv_auroc', 'training_auroc', 'testing_auroc'], var_name='kind', value_name='auroc')
grid = sns.factorplot(y='symbol', x='auroc', hue='kind', data=plot_df, kind="bar")
#xlimits = grid.ax.set_xlim(0.5, 1)


# In[ ]:



