
# coding: utf-8

# # Compare logistic regression models of TP53 mutation for a) covariates only and b) covariates with gene expression data to determine marginal gain using gene expression data 

# In[167]:

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


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Specify model configuration

# In[3]:

# We're going to be building a 'TP53' classifier 
GENE = '7157' # TP53


# *Here is some [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) regarding the classifier and hyperparameters*
# 
# *Here is some [information](https://ghr.nlm.nih.gov/gene/TP53) about TP53*

# ## Load Data

# In[4]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', '..', 'download', 'covariates.tsv')\ncovariates = pd.read_table(path, index_col=0)\npath = os.path.join('..', '..', 'download', 'expression-matrix.tsv.bz2')\nexpression = pd.read_table(path, index_col=0)")


# In[14]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..','..','download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[278]:

X = {}
X['a'] = covariates
X['b'] = pd.concat([covariates,expression], axis=1)
for k in ['a','b']:
    print(X[k].shape)


# In[147]:

# Which columns have NaN values? 
for k in ['a','b']:
    print(X[k].columns[pd.isnull(X[k]).any()].tolist())


# In[159]:

y = Y[GENE]


# In[160]:

# The Series now holds TP53 Mutation Status for each Sample
y.head(6)


# In[164]:

# Here are the percentage of tumors with NF1
y.value_counts(True)


# ## Set aside 10% of the data for testing

# In[165]:

X_train = {}
X_test = {}
for k in ['a','b']:
    X_train[k], X_test[k], y_train, y_test = train_test_split(X[k], y, test_size=0.1, random_state=0)
    'Size: {:,} features, {:,} training samples, {:,} testing samples'.format(len(X[k].columns), 
                                                                              len(X_train[k]), len(X_test[k]))


# ## Median absolute deviation feature selection

# In[20]:

def fs_mad(x, y):
    """    
    Get the median absolute deviation (MAD) for each column of x
    """
    scores = mad(x) 
    return scores, np.array([np.NaN]*len(scores))


# ## Define pipeline and Cross validation model fitting

# In[22]:

# Parameter Sweep for Hyperparameters
param_grid = {
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline = Pipeline(steps=[
    ('imputer', Imputer()),    
    ('select', SelectKBest(fs_mad)),
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])

cv_pipeline = {}
for k in ['a','b']:
    if k == 'a': param_grid['select__k'] = ['all']
    elif k=='b': param_grid['select__k'] = [2000]
    cv_pipeline[k] =  GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc')


# In[62]:

get_ipython().run_cell_magic('time', '', "# Train model a.\ncv_pipeline['a'].fit(X=X_train['a'], y=y_train)")


# In[23]:

get_ipython().run_cell_magic('time', '', "# Train model b.\ncv_pipeline['b'].fit(X=X_train['b'], y=y_train)")


# In[168]:

# Best Params
for k in ['a','b']:
    print('{:.3%}'.format(cv_pipeline[k].best_score_))
    print(cv_pipeline[k].best_params_)


# ## Visualize hyperparameters performance

# In[169]:

cv_result_df = {}
for k in ['a','b']:
    cv_result_df[k] = pd.concat([
        pd.DataFrame(cv_pipeline[k].cv_results_),
        pd.DataFrame.from_records(cv_pipeline[k].cv_results_['params']),
    ], axis='columns')
    display(cv_result_df[k].head(2))


# In[170]:

# Cross-validated performance heatmap
for i,k in enumerate(['a','b']):
    ax = plt.subplot(2,1,i+1)
    cv_score_mat = pd.pivot_table(cv_result_df[k], values='mean_test_score', index='classify__l1_ratio', columns='classify__alpha')

    ax = sns.heatmap(cv_score_mat, annot=True, fmt='.1%')
    if i == 1: ax.set_xlabel('Regularization strength multiplier (alpha)')
    else: ax.set_xlabel('')
    ax.set_ylabel('Elastic net mix param (l1_ratio)');


# ## Use Optimal Hyperparameters to Output ROC Curve

# In[180]:

def get_threshold_metrics(y_true, y_pred):
    roc_columns = ['fpr', 'tpr', 'threshold']
    roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
    roc_df = pd.DataFrame.from_items(roc_items)
    auroc = roc_auc_score(y_true, y_pred)
    return {'auroc': auroc, 'roc_df': roc_df}

y_pred_train = {}
y_pred_test = {}
metrics_train = {}
metrics_test = {}
for k in ['a','b']:
    y_pred_train[k] = cv_pipeline[k].decision_function(X_train[k])
    y_pred_test[k] = cv_pipeline[k].decision_function(X_test[k])

    metrics_train[k] = get_threshold_metrics(y_train, y_pred_train[k])
    metrics_test[k] = get_threshold_metrics(y_test, y_pred_test[k])


# In[186]:

# Plot ROC
for i,k in enumerate(['a','b']):
    for label, metrics in ('Training', metrics_train[k]), ('Testing', metrics_test[k]):
        roc_df = metrics['roc_df']
        plt.plot(roc_df.fpr, roc_df.tpr,
            label='{} (AUROC = {:.1%})'.format(label+'_'+str(k), metrics['auroc']))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Predicting TP53 mutation from gene expression (ROC curves)')
    plt.legend(loc='lower right');


# ## What are the classifier coefficients?

# In[188]:

final_pipeline = {}
final_classifier = {}
for k in ['a','b']:
    final_pipeline[k] = cv_pipeline[k].best_estimator_
    final_classifier[k] = final_pipeline[k].named_steps['classify']


# In[194]:

coef_df = {}
for k in ['a','b']:
    select_indices = final_pipeline[k].named_steps['select'].transform(
        np.arange(len(X[k].columns)).reshape(1, -1)
    ).tolist()

    coef_df[k] = pd.DataFrame.from_items([
        ('feature', X[k].columns[select_indices]),
        ('weight', final_classifier[k].coef_[0]),
    ])

    coef_df[k]['abs'] = coef_df[k]['weight'].abs()
    coef_df[k] = coef_df[k].sort_values('abs', ascending=False)


# In[196]:

for k in ['a','b']:
    print('{:.1%} zero coefficients; {:,} negative and {:,} positive coefficients'.format(
            (coef_df[k].weight == 0).mean(),
            (coef_df[k].weight < 0).sum(),
            (coef_df[k].weight > 0).sum()
    ))


# In[263]:

# What are the top weighted features for model a and model b?
display(coef_df['b'].head(5))
display(coef_df['a'].head(5))


# In[265]:

# What are the model a features in model b?
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

coef_df['b']['isFeatureFromA'] = [not RepresentsInt(x) for x in coef_df['b']['feature']]
display(coef_df['b'].query('isFeatureFromA'))
print('rank of model a features in model b:', np.flatnonzero(coef_df['b']['isFeatureFromA']))


# ## Investigate the predictions

# In[268]:

predict_df = {}
for k in ['a','b']:
    predict_df[k] = pd.DataFrame.from_items([
        ('sample_id', X[k].index),
        ('testing', X[k].index.isin(X_test[k].index).astype(int)),
        ('status', y),
        ('decision_function', cv_pipeline[k].decision_function(X[k])),
        ('probability', cv_pipeline[k].predict_proba(X[k])[:, 1]),
    ])
    predict_df[k]['probability_str'] = predict_df[k]['probability'].apply('{:.1%}'.format)


# In[269]:

# Top predictions amongst negatives (potential hidden responders)
for k in ['a','b']:
    display(predict_df[k].sort_values('decision_function', ascending=False).query("status == 0").head(5))


# In[276]:

# Ignore numpy warning caused by seaborn
warnings.filterwarnings('ignore', 'using a non-integer number instead of an integer')
for k in ['a','b']:
    ax = sns.distplot(predict_df[k].query("status == 0").decision_function, hist=False, label=k+' Negatives')
    ax = sns.distplot(predict_df[k].query("status == 1").decision_function, hist=False, label=k+' Positives')


# In[277]:

for k in ['a','b']:
    ax = sns.distplot(predict_df[k].query("status == 0").probability, hist=False, label=k+' Negatives')
    ax = sns.distplot(predict_df[k].query("status == 1").probability, hist=False, label=k+' Positives')


# In[ ]:



