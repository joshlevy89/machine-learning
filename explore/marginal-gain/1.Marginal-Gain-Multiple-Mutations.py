
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


# In[8]:

mutations = {
    '5979': 'RET'    # ret proto-oncogene   
}


# ## Define pipeline and Cross validation model fitting

# In[29]:

# Parameter Sweep for Hyperparameters

param_grid = {
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [.1,1],#[10 ** x for x in range(-4, 2)],
    'classify__l1_ratio':  [.2]# [0, 0.05, 0.1, 0.2, 0.5],
}


pipeline = Pipeline(steps=[
    ('impute', Imputer()),
    ('variance', VarianceThreshold()),
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])


# In[30]:

cv_pipeline = {}
cv_pipeline['model a'] = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc')
cv_pipeline['model b'] = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc')


# ## Functions to get statistics for a given model 

# In[31]:

# Get statistics for a given model. 

def get_aurocs(X, y, pipeline, series, model_type):
    """
    Fit the classifier for the given mutation (y) and output predictions for it
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    pipeline.fit(X=X_train, y=y_train)
    y_pred_train = pipeline.decision_function(X_train)
    y_pred_test = pipeline.decision_function(X_test)
    cv_score_df = grid_scores_to_df(pipeline.grid_scores_)
    coeff_df = get_coeffs(pipeline, X_train, model_type)
    n_pos, n_neg = get_sign_coeffs(coeff_df)
    cov_ranks = get_ranks_covariates_feat(coeff_df)
    series['mean_cv_auroc'] = cv_score_df.score.max()
    series['training_auroc'] = roc_auc_score(y_train, y_pred_train)
    series['testing_auroc'] = roc_auc_score(y_test, y_pred_test)
    series['best_l1_ratio'] = pipeline.best_params_['classify__l1_ratio']
    series['best_alpha'] = pipeline.best_params_['classify__alpha']
    series['n_pos_coeffs'] = n_pos
    series['n_neg_coeffs'] = n_neg
    series['n_positive_mutation'] = sum(y==1)
    series['n_negative_mutation'] = sum(y==0)
    series['cum_rank_cov_feat'] = cov_ranks.sum()
    series['median_rank_cov_feat'] = np.median(cov_ranks)
    series['mean_rank_cov_feat'] = np.mean(cov_ranks)
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

def get_coeffs(pipeline, X_train, model_type):
    """
    Get a dataframe with the training coefficients 
    """
    final_pipeline = pipeline.best_estimator_
    final_classifier = final_pipeline.named_steps['classify']
    
    # Get indices of features
    if model_type == 'model a': 
        #select_indices = list(range(len(covariates.columns.values)))
        select_indices = final_pipeline.named_steps['variance'].transform(
            np.arange(len(X_train.columns)).reshape(1, -1)
        ).tolist()
    else:
#         select_indices = final_pipeline.named_steps['features'].transform(
#             np.arange(len(X_train.columns)).reshape(1, -1)
#         ).tolist()
        select_indices = final_pipeline.named_steps['variance'].transform(
            np.arange(len(X_train.columns)).reshape(1, -1)
        ).tolist()
        
    select_indices = [x for sublist in select_indices for x in sublist]
    
    # Make df features, weights
    coef_df = pd.DataFrame.from_items([
        ('feature', X_train.columns[select_indices]),
        ('weight', final_classifier.coef_[0]),
    ])

    return coef_df

def get_sign_coeffs(coef_df):
    return (coef_df.weight>0).sum(), (coef_df.weight<0).sum()

def get_ranks_covariates_feat(coef_df):
    coef_df['abs'] = coef_df['weight'].abs()
    coef_df = coef_df.sort_values('abs', ascending=False)
    
    def RepresentsInt(s):
        try: 
            int(s)
            return True
        except ValueError:
            return False

    coef_df['is_cov_feat'] = [not RepresentsInt(x) for x in coef_df['feature']]
    ranks = np.flatnonzero(coef_df['is_cov_feat'])
    return ranks

auroc_dfs = {}


# ## Train the models.

# In[32]:

get_ipython().run_cell_magic('time', '', '# Train model a: covariates only.\nwarnings.filterwarnings("ignore") # ignore deprecation warning for grid_scores_\nrows = list()\nfor m in list(mutations):\n    series = pd.Series()\n    series[\'mutation\'] = m\n    series[\'symbol\'] = mutations[m]\n    rows.append(get_aurocs(X[\'model a\'], Y[m], cv_pipeline[\'model a\'], series, \'model a\'))\nauroc_dfs[\'model a\'] = pd.DataFrame(rows)\nauroc_dfs[\'model a\'].sort_values([\'symbol\', \'testing_auroc\'], ascending=[True, False], inplace=True)\ndisplay(auroc_dfs[\'model a\'])')


# In[33]:

get_ipython().run_cell_magic('time', '', '# Train model b: covariates with gene expression data.\nwarnings.filterwarnings("ignore") # ignore deprecation warning for grid_scores_\nrows = list()\nfor m in list(mutations):\n    series = pd.Series()\n    series[\'mutation\'] = m\n    series[\'symbol\'] = mutations[m]\n    rows.append(get_aurocs(X[\'model b\'], Y[m], cv_pipeline[\'model b\'], series, \'model b\'))\nauroc_dfs[\'model b\'] = pd.DataFrame(rows)\nauroc_dfs[\'model b\'].sort_values([\'symbol\', \'testing_auroc\'], ascending=[True, False], inplace=True)\ndisplay(auroc_dfs[\'model b\'])')


# In[34]:

auroc_dfs['model a']['model'] = 'covariates_only'
auroc_dfs['model b']['model'] = 'combined'
auroc_df = pd.concat([auroc_dfs['model a'],auroc_dfs['model b']])
auroc_df.to_csv("./auroc_df.tsv", sep="\t", float_format="%.3g", index=False)
display(auroc_df)


# In[15]:

auroc_dfs['model a'] = auroc_dfs['model a'].drop('model',axis=1)
auroc_dfs['model b'] = auroc_dfs['model b'].drop('model',axis=1)
auroc_dfs['diff_models_ab'] = auroc_dfs['model b'].loc[:,'mean_cv_auroc':]-auroc_dfs['model a'].loc[:,'mean_cv_auroc':]
auroc_dfs['diff_models_ab'][['mutation', 'symbol']] = auroc_dfs['model b'].loc[:, ['mutation', 'symbol']]
auroc_dfs['diff_models_ab']


# # Covariates only vs covariates+expression model

# In[16]:

plot_df = pd.melt(auroc_dfs['diff_models_ab'], id_vars='symbol', value_vars=['mean_cv_auroc', 'training_auroc', 'testing_auroc'], var_name='kind', value_name='delta auroc')
grid = sns.factorplot(y='symbol', x='delta auroc', hue='kind', data=plot_df, kind="bar")

