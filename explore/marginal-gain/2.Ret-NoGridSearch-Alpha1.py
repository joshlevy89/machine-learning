
# coding: utf-8

# # Compare logistic regression models of several mutations for a) covariates only and b) covariates with gene expression data to determine marginal gain using gene expression data 

# In[2]:

import os
import urllib
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, grid_search
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer, FunctionTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from statsmodels.robust.scale import mad
from IPython.display import display
import gc


# In[3]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Load Data

# In[4]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', '..', 'download', 'covariates.tsv')\ncovariates = pd.read_table(path, index_col=0)")


# In[5]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', '..', 'download', 'expression-matrix.tsv.bz2')\nexpression = pd.read_table(path, index_col=0)")


# In[6]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..','..','download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[7]:

X = {}
X['model a'] = covariates
X['model b'] = pd.concat([covariates,expression], axis=1)
for k in ['model a','model b']:
    print(X[k].shape)


# In[38]:

mutations = {
    '5979': 'RET'    # ret proto-oncogene   
}


# ## Define pipeline and Cross validation model fitting

# In[47]:

pipeline = Pipeline(steps=[
    ('impute', Imputer()),
    ('variance', VarianceThreshold()),
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, loss='log', penalty='elasticnet', 
                               l1_ratio=.2, alpha=1, class_weight='balanced'))
])


# ## Functions to get statistics for a given model 

# In[48]:

# Get statistics for a given model. 

def get_aurocs(X, y, pipeline, series):
    """
    Fit the classifier for the given mutation (y) and output predictions for it
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    pipeline.fit(X=X_train, y=y_train)
    y_pred_train = pipeline.decision_function(X_train)
    y_pred_test = pipeline.decision_function(X_test)
    series['training_auroc'] = roc_auc_score(y_train, y_pred_train)
    series['testing_auroc'] = roc_auc_score(y_test, y_pred_test)
    return series 

auroc_dfs = {}


# ## Train the models.

# In[49]:

get_ipython().run_cell_magic('time', '', '# Train model a: covariates only.\nwarnings.filterwarnings("ignore") # ignore deprecation warning for grid_scores_\nrows = list()\nfor m in list(mutations):\n    series = pd.Series()\n    series[\'mutation\'] = m\n    series[\'symbol\'] = mutations[m]\n    rows.append(get_aurocs(X[\'model a\'], Y[m], pipeline, series))\nauroc_dfs[\'model a\'] = pd.DataFrame(rows)\nauroc_dfs[\'model a\'].sort_values([\'symbol\', \'testing_auroc\'], ascending=[True, False], inplace=True)\ndisplay(auroc_dfs[\'model a\'])')


# In[51]:

get_ipython().run_cell_magic('time', '', '# Train model b: covariates with gene expression data.\nwarnings.filterwarnings("ignore") # ignore deprecation warning for grid_scores_\nrows = list()\nfor m in list(mutations):\n    series = pd.Series()\n    series[\'mutation\'] = m\n    series[\'symbol\'] = mutations[m]\n    rows.append(get_aurocs(X[\'model b\'], Y[m], pipeline, series))\nauroc_dfs[\'model b\'] = pd.DataFrame(rows)\nauroc_dfs[\'model b\'].sort_values([\'symbol\', \'testing_auroc\'], ascending=[True, False], inplace=True)\ndisplay(auroc_dfs[\'model b\'])')


# In[52]:

auroc_dfs['model a']['model'] = 'covariates_only'
auroc_dfs['model b']['model'] = 'combined'
auroc_df = pd.concat([auroc_dfs['model a'],auroc_dfs['model b']])
auroc_df.to_csv("./auroc_df.tsv", sep="\t", float_format="%.3g", index=False)
display(auroc_df)


# In[53]:

auroc_dfs['model a'] = auroc_dfs['model a'].drop('model',axis=1)
auroc_dfs['model b'] = auroc_dfs['model b'].drop('model',axis=1)
auroc_dfs['diff_models_ab'] = auroc_dfs['model b'].loc[:,'mean_cv_auroc':]-auroc_dfs['model a'].loc[:,'mean_cv_auroc':]
auroc_dfs['diff_models_ab'][['mutation', 'symbol']] = auroc_dfs['model b'].loc[:, ['mutation', 'symbol']]
auroc_dfs['diff_models_ab']


# # Covariates only vs covariates+expression model

# In[ ]:

plot_df = pd.melt(auroc_dfs['diff_models_ab'], id_vars='symbol', value_vars=['mean_cv_auroc', 'training_auroc', 'testing_auroc'], var_name='kind', value_name='delta auroc')
grid = sns.factorplot(y='symbol', x='delta auroc', hue='kind', data=plot_df, kind="bar")


# In[ ]:



