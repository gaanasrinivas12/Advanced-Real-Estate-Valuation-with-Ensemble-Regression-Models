#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# ### Features and the target

# In[4]:


X=train.copy()
y=X.pop("SalePrice")


# In[ ]:


X.head(5)


# Here, I drop Id because it is irrelevant

# In[5]:


X.drop('Id',axis=1,inplace=True)


# # Exploratory Data Analysis

# In[ ]:


X.dtypes.to_frame()


# In[ ]:


X.isnull().sum().to_frame()


# In[ ]:


X.describe()


# In[ ]:


import scipy.stats as stats

sns.set(style="whitegrid", palette="pastel")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3)

ax0 = fig.add_subplot(gs[1, :])
sns.histplot(y, bins=100, kde=True, color="#18b29d", ax=ax0, alpha=0.7)
ax0.set_title('Distribution of Sale Price', fontsize=18, fontweight='bold', color="#2c3e50")
ax0.set_xlabel('Sale Price', fontsize=14, fontweight='bold', color="#2c3e50")
ax0.set_ylabel('Frequency', fontsize=14, fontweight='bold', color="#2c3e50")
ax0.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
ax0.tick_params(axis='x', labelsize=12, colors="#2c3e50")
ax0.tick_params(axis='y', labelsize=12, colors="#2c3e50")

ax1 = fig.add_subplot(gs[0, 0])
stats.probplot(y, dist="norm", plot=ax1)
ax1.get_lines()[1].set_color("#e74c3c")
ax1.get_lines()[1].set_alpha(0.7)
ax1.set_title('QQ Plot (Normal Distribution)', fontsize=16, fontweight='bold', color="#2c3e50")
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
ax1.tick_params(axis='x', labelsize=12, colors="#2c3e50")
ax1.tick_params(axis='y', labelsize=12, colors="#2c3e50")

ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(y, dist="expon", plot=ax2)
ax2.get_lines()[1].set_color("#e74c3c")
ax2.get_lines()[1].set_alpha(0.7)
ax2.set_title('QQ Plot (Exponential Distribution)', fontsize=16, fontweight='bold', color="#2c3e50")
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
ax2.tick_params(axis='x', labelsize=12, colors="#2c3e50")
ax2.tick_params(axis='y', labelsize=12, colors="#2c3e50")

ax3 = fig.add_subplot(gs[0, 2])
stats.probplot(np.log(y), dist="norm", plot=ax3)
ax3.get_lines()[1].set_color("#e74c3c")
ax3.get_lines()[1].set_alpha(0.7)
ax3.set_title('QQ Plot (Log-Normal Distribution)', fontsize=16, fontweight='bold', color="#2c3e50")
ax3.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
ax3.tick_params(axis='x', labelsize=12, colors="#2c3e50")
ax3.tick_params(axis='y', labelsize=12, colors="#2c3e50")

plt.tight_layout()
plt.show()


# In[ ]:


sns.set(style="whitegrid", palette="magma")

df_num_filtered = X.select_dtypes(include=["int64", "float64"])

num_cols = 3
num_features = len(df_num_filtered.columns)
num_rows = (num_features + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 5))
axes = axes.flatten()

for i, col in enumerate(df_num_filtered.columns):
    sns.histplot(df_num_filtered[col], kde=True, color=sns.color_palette()[i % len(sns.color_palette())], ax=axes[i])
    axes[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold', color="#2c3e50")
    axes[i].set_xlabel(col, fontsize=12, fontweight='bold', color="#2c3e50")
    axes[i].set_ylabel('Frequency', fontsize=12, fontweight='bold', color="#2c3e50")
    axes[i].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    axes[i].tick_params(axis='x', labelsize=10, colors="#2c3e50")
    axes[i].tick_params(axis='y', labelsize=10, colors="#2c3e50")
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


# ## MI Scores

# In[ ]:


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    X=X.fillna(0)
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


# In[ ]:


mi_scores=make_mi_scores(X,y)


# In[ ]:


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=False)
    sns.set(style="ticks", palette="magma")
    plt.figure(figsize=(12, 14))
    sns.barplot(x=scores.values, y=scores.index, palette="magma", edgecolor="black", alpha=0.8)
    plt.title("Mutual Information Scores", fontsize=18, fontweight='bold', color="#2c3e50")
    plt.xlabel("MI Score", fontsize=14, fontweight='bold', color="#2c3e50")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', axis='x')
    plt.tight_layout()
    plt.show()

plot_mi_scores(mi_scores)


# In[ ]:


def plot_categorical_columns(data, columns_to_plot):
    sns.set(style="whitegrid", palette="pastel")
    plt.rcParams['figure.figsize'] = (14, 6)

    for column in columns_to_plot:
        plt.figure()
        sns.countplot(x=column, data=data, palette="magma", edgecolor="black", alpha=0.8)

        plt.title(f'Distribution of {column} Categories', fontsize=18, fontweight='bold', color="#2c3e50")
        plt.xlabel(column, fontsize=14, fontweight='bold', color="#2c3e50")
        plt.ylabel('Count', fontsize=14, fontweight='bold', color="#2c3e50")

        plt.xticks(rotation=90, fontsize=12, color="#2c3e50", ha='right')
        plt.yticks(fontsize=12, color="#2c3e50")

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', axis='y')
        plt.tight_layout()
        plt.show()

columns_to_plot = ['OverallQual', 'Neighborhood']
plot_categorical_columns(X, columns_to_plot)
sns.reset_orig()


# In[ ]:


def plot_categorical_columns(data, columns_to_plot):
    sns.set(style="ticks", palette="pastel")
    plt.rcParams['figure.figsize'] = (14, 6)

    for column in columns_to_plot:
        plt.figure()
        sns.countplot(x=column, data=data, palette="magma", edgecolor="black", alpha=0.8)

        plt.title(f'Distribution of {column} Categories', fontsize=18, fontweight='bold', color="#2c3e50")
        plt.xlabel(column, fontsize=14, fontweight='bold', color="#2c3e50")
        plt.ylabel('Count', fontsize=14, fontweight='bold', color="#2c3e50")

        plt.xticks(rotation=90, fontsize=12, color="#2c3e50", ha='right')
        plt.yticks(fontsize=12, color="#2c3e50")

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', axis='y')
        plt.tight_layout()
        plt.show()

columns_to_plot = ['OverallQual', 'Neighborhood']
plot_categorical_columns(X, columns_to_plot)


# In[ ]:


def plot_missing_data_heatmaps(X):
    X_num = X.select_dtypes(include=["int64","float64"])
    X_obj = X.drop(columns=X_num)

    heatmap_dfs = [X_num, X_obj]
    titles = ["Numeric Features", "Categorical Features"]
    sns.reset_orig()

    for df, title in zip(heatmap_dfs, titles):
        plt.figure(figsize=(14, 6))
        ax = sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.xticks(rotation=90)
        ax.xaxis.set_tick_params(which='both', length=5, width=1.5, color='black')

        plt.title(f'Missing Data Heatmap: {title}', fontsize=16, fontweight='bold', color="#2c3e50")
        plt.tight_layout()
        plt.show()

plot_missing_data_heatmaps(X)


# In[ ]:


columns_to_plot = ['BsmtQual']
plot_categorical_columns(X, columns_to_plot)


# In[ ]:


def plot_boxplot(X, x, y):
    plt.figure(figsize=(14, 6))

    sns.boxplot(x=x, y=y, data=X, palette="coolwarm",
                linewidth=2.5, fliersize=5, boxprops=dict(edgecolor="black"),
                medianprops=dict(color="red", linewidth=2))
    plt.title(f'Boxplot of {x} vs. {y}', fontsize=18, fontweight='bold', color="#2c3e50")
    plt.xlabel(x, fontsize=14, fontweight='bold', color="#2c3e50")
    plt.ylabel(y, fontsize=14, fontweight='bold', color="#2c3e50")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.show()

plot_boxplot(X, "OverallQual", "BsmtQual")


# In[ ]:


def plot_scatterplot(X, x, y, z):
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x=x, y=y, hue=z, data=X,
                    palette="magma", s=100, alpha=0.8, edgecolor="black", linewidth=1)

    plt.title(f'Scatterplot of {x} vs. {y}', fontsize=20, fontweight='bold', color="#2c3e50")
    plt.xlabel(x, fontsize=16, fontweight='bold', color="#2c3e50")
    plt.ylabel(y, fontsize=16, fontweight='bold', color="#2c3e50")

    plt.legend(title=z, title_fontsize='13', fontsize='11', loc='best', frameon=True, facecolor='white', edgecolor='black')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.show()

plot_scatterplot(X, "LotFrontage", "LotArea", "Neighborhood")


# ### Chi-square value & P-value

# In[ ]:


from scipy.stats import chi2_contingency

frequency_table = pd.crosstab(X['Neighborhood'], X['MasVnrType'])
chi2, p, _, _ = chi2_contingency(frequency_table)
print(f"Chi-square value: {chi2}")
print(f"P-value: {p}")


# ### Bar Chart

# In[ ]:


def plot_barplot(y, hue, X):
    plt.figure(figsize=(20, 15))
    sns.countplot(y=y, hue=hue, data=X, palette="magma", edgecolor="black", linewidth=1.5)

    plt.title(f'Stacked Bar Plot of {hue} by {y}', fontsize=22, fontweight='bold', color="#2c3e50")
    plt.xlabel('Count', fontsize=18, fontweight='bold', color="#2c3e50")
    plt.ylabel(y, fontsize=18, fontweight='bold', color="#2c3e50")

    plt.yticks(rotation=0, fontsize=14, color="#2c3e50")
    plt.xticks(fontsize=14, color="#2c3e50")

    plt.legend(title=hue, title_fontsize='16', fontsize='14', loc='best', frameon=True, facecolor='white', edgecolor='black')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', axis='x')
    plt.tight_layout()
    plt.show()

hue = ['MasVnrType','BsmtExposure']
for hue in hue:
    plot_barplot('Neighborhood', hue, X)


# ## Heatmap of MasVnrType vs. Neighborhood

# In[ ]:


def plot_heatmap(frequency_table, title):
    plt.figure(figsize=(16, 8))
    heatmap = sns.heatmap(frequency_table, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt='g',
                          cbar=True, linewidths=0.5, linecolor='white',
                          annot_kws={"size": 12, "weight": "bold"})

    plt.title(title, fontsize=22, fontweight='bold', color="#2c3e50", pad=20)
    plt.xlabel('Neighborhood', fontsize=16, fontweight='bold', color="#2c3e50", labelpad=10)
    plt.ylabel('MasVnrType', fontsize=16, fontweight='bold', color="#2c3e50", labelpad=10)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12, colors="#2c3e50")
    cbar.set_label('Frequency', fontsize=14, fontweight='bold', color="#2c3e50", labelpad=10)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

plot_heatmap(frequency_table=frequency_table, title='Heatmap of MasVnrType vs. Neighborhood')
sns.reset_orig()


# ### Correlation between LotArea and LotFrontage

# In[ ]:


correlation_coefficient=X["LotArea"].corr(X["LotFrontage"])
correlation_coefficient


# In[ ]:


correlation_coefficient=X["OverallQual"].corr(X["OverallCond"])
correlation_coefficient


# In[ ]:


plot_barplot('OverallCond','BsmtExposure',X)


# In[ ]:


plot_categorical_columns(X, ['GarageType'])


# In[ ]:


plot_categorical_columns(X, ['GarageFinish'])


# # Data Preprocessing

# In[6]:


features_drop = ['Alley','MasVnrType','FireplaceQu','PoolQC','Fence','MiscFeature']
X.drop(features_drop,axis=1,inplace=True)


# In[7]:


X_num = X.select_dtypes(include=["int64","float64"])
X_obj = X.drop(columns=X_num)


# In[8]:


X_num.apply(lambda x: x.nunique()).to_frame()


# ## Numerical Features

# In[9]:


Neighborhood_Lot=X.groupby("Neighborhood")["LotFrontage"].mean()
X_num["LotFrontage"]=X["LotFrontage"].fillna(X["Neighborhood"].map(Neighborhood_Lot))


# In[10]:


X["GarageYrBlt"] = X["GarageYrBlt"].fillna(0)
X_num["GarageYrBlt"] = X["GarageYrBlt"]


# In[11]:


Neighborhood_MasVnrArea=X.groupby("Neighborhood")["MasVnrArea"].mean()
X["MasVnrArea"]=X["MasVnrArea"].fillna(X["Neighborhood"].map(Neighborhood_MasVnrArea))
X_num["MasVnrArea"] = X["MasVnrArea"]


# In[12]:


X_num = X_num.fillna(X_num.mean())


# ## Categorical Features

# In[13]:


X.loc[X["OverallQual"]>=8,"BsmtCond"]=X.loc[X["OverallQual"]>=8,"BsmtQual"].fillna("Ex")
X.loc[X["OverallQual"]>=6,"BsmtCond"]=X.loc[X["OverallQual"]>=6,"BsmtQual"].fillna("Gd")
X.loc[X["OverallQual"]>=5,"BsmtCond"]=X.loc[X["OverallQual"]>=5,"BsmtQual"].fillna("TA")
X.loc[X["OverallQual"]>=4,"BsmtCond"]=X.loc[X["OverallQual"]>=4,"BsmtQual"].fillna("Fa")
X.loc[X["OverallQual"]<=4,"BsmtCond"]=X.loc[X["OverallQual"]<=4,"BsmtQual"].fillna("Po")
X_obj["BsmtCond"] = X['BsmtCond']


# In[14]:


X.loc[X["OverallQual"]>=8,"BsmtQual"].fillna("Ex", inplace=True)
X.loc[(X["OverallQual"] >= 6) & (X["OverallQual"] < 8), "BsmtQual"].fillna("Gd", inplace=True)
X.loc[(X["OverallQual"] >= 4) & (X["OverallQual"] < 6), "BsmtQual"].fillna("Fa", inplace=True)
X.loc[X["OverallQual"]<4,"BsmtQual"].fillna("Fa", inplace=True)
X_obj["BsmtQual"] =  X["BsmtQual"]


# In[15]:


X["BsmtExposure"].fillna("No", inplace=True)
X_obj['BsmtExposure'] = X['BsmtExposure']


# In[16]:


cond_fintype1=X.groupby("OverallCond")["BsmtFinType1"].transform(lambda x: x.mode().iloc[0])
X["BsmtFinType1"].fillna(cond_fintype1, inplace=True)
X_obj["BsmtFinType1"] = X["BsmtFinType1"]
cond_fintype2=X.groupby("OverallCond")["BsmtFinType2"].transform(lambda x: x.mode().iloc[0])
X["BsmtFinType2"].fillna(cond_fintype2, inplace=True)
X_obj["BsmtFinType2"] = X["BsmtFinType2"]


# In[17]:


X["Electrical"].fillna("SBrkr", inplace=True)
X_obj["Electrical"] = X["Electrical"]


# In[18]:


X["GarageType"].fillna("NG", inplace=True)
X_obj["GarageType"] = X["GarageType"]
X["GarageFinish"].fillna("NG", inplace=True)
X_obj["GarageFinish"] = X["GarageFinish"]
X["GarageQual"].fillna("NG", inplace=True)
X_obj["GarageQual"] = X["GarageQual"]
X["GarageCond"].fillna("NG", inplace=True)
X_obj["GarageCond"] = X["GarageCond"]


# In[19]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ordered_cats = ["Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "SaleType", "SaleCondition"]
non_ordered_cats = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Heating", "CentralAir", "Electrical", "Functional", "PavedDrive"]
cols_to_encode = list(set(ordered_cats) | set(non_ordered_cats))
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False)
for cat in ordered_cats:
    X_obj[cat] = label_encoder.fit_transform(X_obj[cat])
one_hot_cats = one_hot_encoder.fit_transform(X_obj[non_ordered_cats])
X_obj_complete = pd.concat([X_obj.drop(non_ordered_cats, axis=1), pd.DataFrame(one_hot_cats, columns=one_hot_encoder.get_feature_names_out(non_ordered_cats))], axis=1)


# In[20]:


from scipy import stats

def remove_outliers_zscore(df, columns, threshold=4.0):
    initial_row_count = df.shape[0]
    z_scores = np.abs(stats.zscore(df[columns]))
    df = df[(z_scores < threshold).all(axis=1)]
    final_row_count = df.shape[0]
    rows_removed = initial_row_count - final_row_count
    print(f"Removed {rows_removed} rows due to outliers with Z-score threshold {threshold}.")
    return df

continuous_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                   '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                   'EnclosedPorch','ScreenPorch']

X_num_no_outliers = remove_outliers_zscore(X_num.copy(), continuous_cols)

X_obj_no_outliers = X_obj_complete.loc[X_num_no_outliers.index]
y_no_outliers = y.loc[X_num_no_outliers.index]


# In[21]:


X_complete_no_outliers = pd.concat([X_num_no_outliers, X_obj_no_outliers], axis=1)


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_complete_no_outliers, y_no_outliers, test_size=0.2, random_state=666)


# In[23]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])


# In[24]:


get_ipython().system('pip install optuna catboost')


# # Model Training
# ## Stacked Ensemble
# - Random Forest
# - XGB Boost
# - Light GBM
# - CatBoost
# - Artificial Neural Network
# 
# Linear Regression for the Meta Model

# In[25]:


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[26]:


import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# In[27]:


from sklearn.metrics import make_scorer, mean_squared_log_error

def rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# In[ ]:


def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    return cross_val_score(rf, X_train, y_train, cv=5, scoring=rmsle_scorer).mean()

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(rf_objective, n_trials=50)
best_rf_params = rf_study.best_params
rf = RandomForestRegressor(**best_rf_params)


# In[31]:


def evaluate_model(y_true, y_pred, n, k):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rmsle = np.sqrt(mean_squared_log_error(np.clip(y_true, 0, None), np.clip(y_pred, 0, None)))
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - k - 1))

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Root Mean Squared Logarithmic Error (RMSLE): {rmsle}")
    print(f"R-squared (R²): {r2}")
    print(f"Adjusted R-squared (R²): {adjusted_r2}")

    return {'MSE': mse, 'RMSE': rmse, 'RMSLE': rmsle, 'R²': r2, 'Adjusted R²': adjusted_r2}


# In[32]:


rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
evaluate_model(y_test, rf_y_pred, n=len(y_test), k=X_train.shape[1])


# In[222]:


best_rf_params


# In[ ]:


def xgb_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 0.1, 10.0)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 0.1, 10.0)
    min_child_weight = trial.suggest_float('min_child_weight', 1.0, 10.0)

    xgb = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        random_state=42
    )
    return cross_val_score(xgb, X_train, y_train, cv=5, scoring=rmsle_scorer).mean()


xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=50)
best_xgb_params = xgb_study.best_params
xgb = XGBRegressor(**best_xgb_params)


# In[221]:


best_xgb_params


# In[33]:


xgb.fit(X_train, y_train)
xgb_y_pred = xgb.predict(X_test)
evaluate_model(y_test, xgb_y_pred, n=len(y_test), k=X_train.shape[1])


# In[ ]:


def lgb_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
    num_leaves = trial.suggest_int('num_leaves', 20, 150)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 0.1, 10.0)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 0.1, 10.0)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 50)

    lgb = LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_samples=min_child_samples,
        random_state=42
    )
    return cross_val_score(lgb, X_train, y_train, cv=5, scoring=rmsle_scorer).mean()



lgb_study = optuna.create_study(direction='maximize')
lgb_study.optimize(lgb_objective, n_trials=50)
best_lgb_params = lgb_study.best_params
lgb = LGBMRegressor(**best_lgb_params)


# In[220]:


best_lgb_params


# In[ ]:


lgb.fit(X_train, y_train)
lgb_y_pred = lgb.predict(X_test)
evaluate_model(y_test, lgb_y_pred, n=len(y_test), k=X_train.shape[1])


# In[161]:


def cat_objective(trial):
    bagging_temperature = trial.suggest_float('bagging_temperature', 0.1, 1.0)
    depth = trial.suggest_int('depth', 3, 10)
    iterations = trial.suggest_int('iterations', 100, 1000)
    l2_leaf_reg = trial.suggest_loguniform('l2_leaf_reg', 0.1, 10.0)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)

    cat = CatBoostRegressor(
        bagging_temperature=bagging_temperature,
        depth=depth,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        learning_rate=learning_rate,
        random_state=42,
        verbose=0
    )

    return cross_val_score(cat, X_train, y_train, cv=5, scoring=rmsle_scorer).mean()


cat_study = optuna.create_study(direction='maximize')
cat_study.optimize(cat_objective, n_trials=50)
best_cat_params = cat_study.best_params
cat = CatBoostRegressor(**best_cat_params, verbose=0)


# In[219]:


best_cat_params


# In[36]:


cat.fit(X_train, y_train)
cat_y_pred = cat.predict(X_test)
evaluate_model(y_test, cat_y_pred, n=len(y_test), k=X_train.shape[1])


# In[164]:


def rmsle_keras(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=tf.reduce_max(y_true))
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=tf.reduce_max(y_pred))
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_true) - tf.math.log1p(y_pred))))

def ann_objective(trial):
    def create_ann_model(input_dim):
        model = Sequential()
        model.add(Dense(trial.suggest_int('units1', 64, 512), input_dim=input_dim, activation='relu'))
        model.add(Dense(trial.suggest_int('units2', 32, 256), activation='relu'))
        model.add(Dense(trial.suggest_int('units3', 16, 128), activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(
                          learning_rate=trial.suggest_loguniform('learning_rate', 0.0001, 0.01)),
                      loss=rmsle_keras)  # Use RMSLE as the loss function
        return model

    ann_model = create_ann_model(X_train.shape[1])
    ann_model.fit(X_train, y_train, epochs=trial.suggest_int('epochs', 50, 200),
                  batch_size=trial.suggest_int('batch_size', 16, 64), verbose=0, validation_split=0.1)
    predictions = ann_model.predict(X_train)

    rmsle_score = np.sqrt(mean_squared_log_error(np.clip(y_train, 0, None), np.clip(predictions, 0, None)))
    return rmsle_score

ann_study = optuna.create_study(direction='minimize')
ann_study.optimize(ann_objective, n_trials=50)
best_ann_params = ann_study.best_params


# In[218]:


best_ann_params


# In[165]:


units1 = best_ann_params['units1']
units2 = best_ann_params['units2']
units3 = best_ann_params['units3']
learning_rate = best_ann_params['learning_rate']
epochs = best_ann_params['epochs']
batch_size = best_ann_params['batch_size']

def create_best_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(units1, input_dim=input_dim, activation='relu'))
    model.add(Dense(units2, activation='relu'))
    model.add(Dense(units3, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=rmsle_keras)
    return model

ann_model = create_best_ann_model(X_train.shape[1])
ann_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)


# In[44]:


ann_y_pred = ann_model.predict(X_test)
evaluate_model(y_test, ann_model.predict(X_test), n=len(y_test), k=X_train.shape[1])


# In[172]:


model_preds = {
    "Random Forest": rf_y_pred,
    "XGBoost": xgb_y_pred,
    "LightGBM": lgb_y_pred,
    "CatBoost": cat_y_pred,
    "ANN": ann_y_pred
}

colors = ['deepskyblue', 'salmon', 'limegreen', 'gold', 'purple']
markers = ['o', 's', '^', 'D', 'P']

fig, axs = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Actual vs. Predicted House Prices", fontsize=24, fontweight='bold', color='darkblue')

axs = axs.flatten()

for i, (model, preds) in enumerate(model_preds.items()):
    ax = axs[i]

    ax.scatter(np.exp(y_test / 100000), np.exp(preds / 100000),
               color=colors[i], marker=markers[i], edgecolors="black", alpha=0.8, s=80, label=model)

    ax.plot([0, 8], [0, 8], "orange", lw=2, label="Perfect Prediction Line")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    ax.set_xlabel("Actual Price (in 100,000s)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Predicted Price (in 100,000s)", fontsize=14, fontweight='bold')
    ax.set_title(f"{model} Predictions", fontsize=16, fontweight='bold', color=colors[i])
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)

for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[ ]:


kf = KFold(n_splits=10, shuffle=True, random_state=42)

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

oof_rf = np.zeros(len(X_train_np))
oof_xgb = np.zeros(len(X_train_np))
oof_lgb = np.zeros(len(X_train_np))
oof_cat = np.zeros(len(X_train_np))
oof_ann = np.zeros(len(X_train_np))

X_meta_train = np.zeros((len(X_train_np), 5))
X_meta_test = np.zeros((len(X_test), 5))

for train_idx, val_idx in kf.split(X_train_np):
    X_fold_train, X_fold_val = X_train_np[train_idx], X_train_np[val_idx]
    y_fold_train, y_fold_val = y_train_np[train_idx], y_train_np[val_idx]

    # Random Forest
    rf.fit(X_fold_train, y_fold_train)
    oof_rf[val_idx] = rf.predict(X_fold_val)
    X_meta_test[:, 0] += rf.predict(X_test) / kf.n_splits

    # XGBoost
    xgb.fit(X_fold_train, y_fold_train)
    oof_xgb[val_idx] = xgb.predict(X_fold_val)
    X_meta_test[:, 1] += xgb.predict(X_test) / kf.n_splits

    # LightGBM
    lgb.fit(X_fold_train, y_fold_train)
    oof_lgb[val_idx] = lgb.predict(X_fold_val)
    X_meta_test[:, 2] += lgb.predict(X_test) / kf.n_splits

    # CatBoost
    cat.fit(X_fold_train, y_fold_train)
    oof_cat[val_idx] = cat.predict(X_fold_val)
    X_meta_test[:, 3] += cat.predict(X_test) / kf.n_splits

    # ANN
    ann_model = create_best_ann_model(X_fold_train.shape[1])
    ann_model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size, verbose=0)
    oof_ann[val_idx] = ann_model.predict(X_fold_val).reshape(-1)
    X_meta_test[:, 4] += ann_model.predict(X_test).reshape(-1) / kf.n_splits


X_meta_train[:, 0] = oof_rf
X_meta_train[:, 1] = oof_xgb
X_meta_train[:, 2] = oof_lgb
X_meta_train[:, 3] = oof_cat
X_meta_train[:, 4] = oof_ann


# In[46]:


meta_model = LinearRegression()
meta_model.fit(X_meta_train, y_train)

final_predictions = meta_model.predict(X_meta_test)

mse = mean_squared_error(y_test, final_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, final_predictions)
adjusted_r_squared = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_meta_test.shape[1] - 1)

final_predictions = np.clip(final_predictions, 0, None)
y_test_clipped = np.clip(y_test, 0, None)
rmsle = np.sqrt(mean_squared_log_error(y_test_clipped, final_predictions))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
print(f"Adjusted R-squared: {adjusted_r_squared}")
print(f"Root Mean Squared Logarithmic Error (RMSLE): {rmsle}")


# In[179]:


plt.figure(figsize=(10, 8))
plt.title("Final Stacked Model: Actual vs. Predicted House Prices", fontsize=24, fontweight='bold', color='darkblue')

plt.scatter(np.exp(y_test / 100000), np.exp(final_predictions / 100000),
            color='dodgerblue', marker='o', edgecolors="black", alpha=0.7, s=100, label="Final Stacked Model")

plt.plot([0, 8], [0, 8], "orange", lw=3, label="Perfect Prediction Line")

plt.xlim(0, 8)
plt.ylim(0, 8)
plt.xlabel("Actual Price (in 100,000s)", fontsize=18, fontweight='bold')
plt.ylabel("Predicted Price (in 100,000s)", fontsize=18, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="lower right", fontsize=16)
plt.tight_layout()
plt.show()


# # Text Data Preprocessing

# In[182]:


idx = test['Id']
test.drop('Id',axis=1,inplace=True)
test.drop(features_drop,axis=1,inplace=True)


# In[183]:


test.isnull().sum().to_frame()


# In[184]:


test_num = test.select_dtypes(include=["int64","float64"])
test_obj = test.drop(columns=X_num)


# In[185]:


test_obj.isnull().sum().to_frame()


# In[186]:


Neighborhood_Lot=test.groupby("Neighborhood")["LotFrontage"].mean()
test_num["LotFrontage"]=test["LotFrontage"].fillna(X["Neighborhood"].map(Neighborhood_Lot))

test["GarageYrBlt"] = test["GarageYrBlt"].fillna(0)
test_num["GarageYrBlt"] = test["GarageYrBlt"]

Neighborhood_MasVnrArea=test.groupby("Neighborhood")["MasVnrArea"].mean()
test["MasVnrArea"]=test["MasVnrArea"].fillna(test["Neighborhood"].map(Neighborhood_MasVnrArea))
test_num["MasVnrArea"] = test["MasVnrArea"]

test_num = test_num.fillna(test_num.mean())


# In[187]:


test_num.mean()


# In[188]:


test["BsmtQual"].isnull().sum()


# In[189]:


test.loc[test["OverallQual"]>=8,"BsmtCond"]=test.loc[test["OverallQual"]>=8,"BsmtQual"].fillna("Ex")
test.loc[test["OverallQual"]>=6,"BsmtCond"]=test.loc[test["OverallQual"]>=6,"BsmtQual"].fillna("Gd")
test.loc[test["OverallQual"]>=5,"BsmtCond"]=test.loc[test["OverallQual"]>=5,"BsmtQual"].fillna("TA")
test.loc[test["OverallQual"]>=4,"BsmtCond"]=test.loc[test["OverallQual"]>=4,"BsmtQual"].fillna("Fa")
test.loc[test["OverallQual"]<4,"BsmtCond"]=test.loc[test["OverallQual"]<4,"BsmtQual"].fillna("Po")
test_obj["BsmtCond"] = test['BsmtCond']

test.loc[test["OverallQual"]>=8,"BsmtQual"]=test.loc[test["OverallQual"]>=8,"BsmtQual"].fillna("Ex")
test.loc[(test["OverallQual"] >= 6) & (test["OverallQual"] < 8), "BsmtQual"] = test.loc[(test["OverallQual"] >= 6) & (test["OverallQual"] < 8), "BsmtQual"].fillna("Gd")
test.loc[(test["OverallQual"] >= 4) & (test["OverallQual"] < 6), "BsmtQual"] = test.loc[(test["OverallQual"] >= 4) & (test["OverallQual"] < 6), "BsmtQual"].fillna("Fa")
test.loc[test["OverallQual"]<4,"BsmtQual"]=test.loc[test["OverallQual"]<4,"BsmtQual"].fillna("Fa")
test_obj["BsmtQual"] =  test["BsmtQual"]

test["BsmtExposure"]=test["BsmtExposure"].fillna("No")
test_obj['BsmtExposure'] = test['BsmtExposure']

cond_fintype1=test.groupby("OverallCond")["BsmtFinType1"].transform(lambda x: x.mode().iloc[0])
test["BsmtFinType1"]=test["BsmtFinType1"].fillna(cond_fintype1)
test_obj["BsmtFinType1"] = test["BsmtFinType1"]
cond_fintype2=test.groupby("OverallCond")["BsmtFinType2"].transform(lambda x:x.mode().iloc[0])
test["BsmtFinType2"]=test["BsmtFinType2"].fillna(cond_fintype2)
test_obj["BsmtFinType2"] = test["BsmtFinType2"]

test["Electrical"]=test["Electrical"].fillna("SBrkr")
test_obj["Electrical"] = test["Electrical"]

test["GarageType"]= test["GarageType"].fillna("NG")
test_obj["GarageType"] = test["GarageType"]
test["GarageFinish"]= test["GarageFinish"].fillna("NG")
test_obj["GarageFinish"] = test["GarageFinish"]
test["GarageQual"]= test["GarageQual"].fillna("NG")
test_obj["GarageQual"] = test["GarageQual"]
test["GarageCond"]= test["GarageCond"].fillna("NG")
test_obj["GarageCond"] = test["GarageCond"]


# In[190]:


test_num.dtypes


# In[192]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")
column_names = test_obj.columns.tolist()
imputer.fit(test_obj[column_names])
test_obj_imputed = imputer.transform(test_obj[column_names])
test_obj_imputed = pd.DataFrame(test_obj_imputed, columns=column_names)


# In[193]:


ordered_cols = ["Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "SaleType", "SaleCondition"]
non_ordered_cols = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Heating", "CentralAir", "Electrical", "Functional", "PavedDrive"]
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False)
for col in ordered_cols:
    test_obj_imputed[col] = label_encoder.fit_transform(test_obj_imputed[col])
one_hot_cols = one_hot_encoder.fit_transform(test_obj_imputed[non_ordered_cols])
test_obj_complete = pd.concat([test_obj_imputed.drop(non_ordered_cols, axis=1), pd.DataFrame(one_hot_cols, columns=one_hot_encoder.get_feature_names_out(non_ordered_cols))], axis=1)


# In[194]:


continuous_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch','ScreenPorch']

test_num_scaled = pd.DataFrame(scaler.fit_transform(test_num[continuous_cols]), columns=continuous_cols)
test_num_scaled = pd.concat([test_num_scaled, test_num.drop(continuous_cols, axis=1)], axis=1)


# In[203]:


test_complete = pd.concat([test_num_scaled, test_obj_complete], axis=1)


# In[204]:


missing_cols = set(X_complete_no_outliers) - set(test_complete)
print(list(missing_cols))


# In[205]:


missing_cols = ['Utilities_NoSeWa', 'Electrical_Mix', 'Heating_OthW', 'Heating_Floor']
X_complete_no_outliers[missing_cols].value_counts()


# In[206]:


test_complete[missing_cols] = 0.0


# In[207]:


col_order = X_complete_no_outliers.columns


# In[208]:


test_complete = test_complete.reindex(columns=col_order)


# In[210]:


test_complete.head(50)


# In[212]:


rf_deploy_preds = rf.predict(test_complete)
xgb_deploy_preds = xgb.predict(test_complete)
lgb_deploy_preds = lgb.predict(test_complete)
cat_deploy_preds = cat.predict(test_complete)
ann_deploy_preds = ann_model.predict(test_complete).reshape(-1)


# In[213]:


X_meta_deploy = np.column_stack((
    rf_deploy_preds,
    xgb_deploy_preds,
    lgb_deploy_preds,
    cat_deploy_preds,
    ann_deploy_preds
))


# In[214]:


final_deploy_predictions = meta_model.predict(X_meta_deploy)


# In[217]:


output = pd.DataFrame({'Id': idx, 'SalePrice': final_deploy_predictions})
output.to_csv('submission.csv', index=False)

