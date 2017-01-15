
# coding: utf-8

# # How much predictiveness does gene expression add over covariates for Ret (using mad)? 

# Some notes on the results displayed here:
# 1) Uses Ret as an example
# 2) Analyzes training/testing auroc for 3 data sets
#     -covariates only
#     -expression only
#     -covariates+expression
# 3) Feature selection was first performed on the expression data.
# 4) Using this approach, expression only yields identical training/testing auroc as in Branka's PR #52 for Ret.

# In[1]:

import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from matplotlib import gridspec
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import SelectKBest
from statsmodels.robust.scale import mad

from IPython.core.debugger import Tracer
from IPython.display import display
import warnings
warnings.filterwarnings("ignore") # ignore deprecation warning for grid_scores_


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# In[3]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', '..', 'download', 'covariates.tsv')\ncovariates = pd.read_table(path, index_col=0)")


# In[4]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', '..', 'download', 'expression-matrix.tsv.bz2')\nexpression = pd.read_table(path, index_col=0)")


# In[5]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..','..','download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[6]:

# Pre-process expression data for use later
def fs_mad(x,y):
    """
    Get the median absolute deviation (MAD) for each column of x
    """
    scores = mad(x) 
    return scores, np.array([np.NaN]*len(scores))

n_features_select = 500
expression_select = SelectKBest(fs_mad, k=n_features_select).fit_transform(expression,np.ones(expression.shape[0]))
expression_select = pd.DataFrame(expression_select)
expression_select = expression_select.set_index(expression.index.values)


# In[7]:

# Create combo data set (processed expression + covariates)
combined = pd.concat([covariates,expression_select],axis=1)
combined.shape


# In[8]:

mutations = {
    '5979': 'RET',    # ret proto-oncogene
}


# In[9]:

# Define model
param_fixed = {
    'loss': 'log',
    'penalty': 'elasticnet',
}
param_grid = {
    'classify__alpha': [10 ** x for x in range(-6, 1)],
    'classify__l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1]
}

pipeline = Pipeline(steps=[
    ('impute', Imputer()),
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced', loss=param_fixed['loss'], 
                               penalty=param_fixed['penalty']))
])

pipeline = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc')


# In[10]:

# Helper training/evaluation functions.
def train_and_evaluate(data, pipeline):
    """
    Train each model using grid search, and ouput:
    1) all_best_estimator_aurocs: contains aurocs for mean_cv, train, and test for chosen grid parameters
    2) all_grid_aurocs: contains aurocs for each hyperparameter-fold combo in grid search
    """
    all_best_estimator_aurocs = list()
    all_grid_aurocs = pd.DataFrame()
    for m in list(mutations):
        best_estimator_aurocs, grid_aurocs = get_aurocs(data, Y[m], pipeline)
        best_estimator_aurocs['symbol'] = mutations[m]
        grid_aurocs['symbol'] = mutations[m]
        all_grid_aurocs = all_grid_aurocs.append(grid_aurocs, ignore_index = True)
        all_best_estimator_aurocs.append(best_estimator_aurocs) 
    all_best_estimator_aurocs = pd.DataFrame(all_best_estimator_aurocs)
    return all_best_estimator_aurocs, all_grid_aurocs

def get_aurocs(X, y, pipeline):
    """
    Fit the classifier for the given mutation (y) and output predictions for it
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    pipeline.fit(X=X_train, y=y_train)
    y_pred_train = pipeline.decision_function(X_train)
    y_pred_test = pipeline.decision_function(X_test)
    grid_aurocs = grid_scores_to_df(pipeline.grid_scores_) 
    best_estimator_aurocs = pd.Series()
    best_estimator_aurocs['mean_cv_auroc'] = grid_aurocs['fold_mean'].max()
    best_estimator_aurocs['training_auroc'] = roc_auc_score(y_train, y_pred_train)
    best_estimator_aurocs['testing_auroc'] = roc_auc_score(y_test, y_pred_test)
    best_estimator_aurocs['best_alpha'] = pipeline.best_params_['classify__alpha']
    best_estimator_aurocs['best_11_ratio'] = pipeline.best_params_['classify__l1_ratio']
    best_estimator_aurocs['n_positive_mutation'] = sum(y==1)
    best_estimator_aurocs['n_negative_mutation'] = sum(y==0)
    return best_estimator_aurocs, grid_aurocs

def grid_scores_to_df(grid_scores):
    """
    Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to 
    a tidy pandas DataFrame where each row is a hyperparameter and each column a fold.
    """
    rows = []
    for grid_score in grid_scores:
        row = np.concatenate(([grid_score.parameters['classify__alpha']],
                              [grid_score.parameters['classify__l1_ratio']],
                            grid_score.cv_validation_scores))
        rows.append(row)
    grid_aurocs = pd.DataFrame(rows,columns=['alpha', 'l1_ratio','fold_1','fold_2','fold_3'])
    grid_aurocs['fold_mean'] = grid_aurocs.iloc[:,2:4].mean(axis=1)
    grid_aurocs['fold_std'] = grid_aurocs.iloc[:,2:4].std(axis=1)
    return grid_aurocs


# In[30]:

# Helper visualization functions.
def visualize_grid_aurocs(grid_aurocs, gene_type=None, ax=None):
    """
    Visualize grid search results for each mutation-alpha parameter combo.
    """
    if ax==None: f, ax = plt.subplots()
    grid_aurocs_mat = pd.pivot_table(grid_aurocs, values='fold_mean', index='l1_ratio', columns='alpha')
    sns.heatmap(grid_aurocs_mat, annot=True, fmt='.2', ax=ax)
    ax.set_ylabel('l1_ratio')
    ax.set_xlabel('Regularization strength multiplier (alpha)')
    plt.setp(ax.get_yticklabels(), rotation=0)
    if gene_type != None: ax.set_title(gene_type, fontsize=15)

def visualize_best_estimator_aurocs(estimator_aurocs, gene_type=None, ax=None, training_data_type=None):
    """
    Creates a bar plot of mean_cv_auroc, training_auroc, and testing_auroc for each gene in df
    """
    plot_df = pd.melt(estimator_aurocs, id_vars='symbol', value_vars=['mean_cv_auroc', 'training_auroc', 
                                                'testing_auroc'], var_name='kind', value_name='aurocs')
    ax = sns.barplot(y='symbol', x='aurocs', hue='kind', data=plot_df,ax=ax)
    if training_data_type == 'marginal_gain': ax.set(xlabel='delta aurocs')
    else: ax.set(xlabel='aurocs')
    ax.legend(bbox_to_anchor=(.65, 1.1), loc=2, borderaxespad=0.)
    plt.setp(ax.get_yticklabels(), rotation=0)
    if gene_type != None: ax.set_title(gene_type, fontsize=15)


# In[12]:

# Train with covariates data
all_best_estimator_aurocs_covariates, all_grid_aurocs_covariates = train_and_evaluate(covariates, pipeline)


# In[26]:

# Visualize covariates data
visualize_best_estimator_aurocs(all_best_estimator_aurocs_covariates, training_data_type='covariates')
visualize_grid_aurocs(all_grid_aurocs_covariates)


# In[14]:

# Train expression data
all_best_estimator_aurocs_expression, all_grid_aurocs_expression = train_and_evaluate(expression_select, pipeline)


# In[27]:

# Visualize expression data
visualize_best_estimator_aurocs(all_best_estimator_aurocs_expression, training_data_type='expression')
visualize_grid_aurocs(all_grid_aurocs_expression)


# In[16]:

# Train with combined data
all_best_estimator_aurocs_combined, all_grid_aurocs_combined = train_and_evaluate(combined, pipeline)


# In[28]:

# Visualize combined data
visualize_best_estimator_aurocs(all_best_estimator_aurocs_combined, training_data_type='expression')
visualize_grid_aurocs(all_grid_aurocs_combined)


# In[31]:

# Display difference in auroc between combined and covariates only 
diff_aurocs = all_best_estimator_aurocs_combined.iloc[:,0:3] - all_best_estimator_aurocs_covariates.iloc[:,0:3]
diff_aurocs['symbol'] = all_best_estimator_aurocs_combined.iloc[:,-1]
visualize_best_estimator_aurocs(diff_aurocs, training_data_type='marginal_gain')

