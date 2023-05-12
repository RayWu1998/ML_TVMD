import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
# 利用GridSearchCV选择最优参数
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import plot_roc_curve
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from collections import Counter

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from mlxtend.regressor import StackingRegressor
import xgboost as xgb
import joblib

# 提取数据
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('./data/opensees_design_res.csv')
df.info()
df['ok'] = True
df.loc[df["drift_max"] > 1 / 200, ['ok']] = False
df.loc[df["a_max"] > 0.35 * 9.8, ['ok']] = False

# 分类问题
# 特征过程
x_label = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9",
           "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "sa1", "sa2", "sa3", "sa4", "sa5", "sa6", "sa7", "sa8",
           "sa9", "sa10", "sa11", "sa12", "sa13", "sa14", "sa15", "sa16", "sa17", "sa18", "sa19", "sa20", "sa21",
           "sa22", "sa23", "sa24", "sa25", "sa26", "sa27", "sa28", "sa29", "sa30", "sv1", "sv2", "sv3", "sv4", "sv5",
           "sv6", "sv7", "sv8", "sv9", "sv10", "sv11", "sv12", "sv13", "sv14", "sv15", "sv16", "sv17", "sv18", "sv19",
           "sv20", "sv21", "sv22", "sv23", "sv24", "sv25", "sv26", "sv27", "sv28", "sv29", "sv30", "sd1", "sd2", "sd3",
           "sd4", "sd5", "sd6", "sd7", "sd8", "sd9", "sd10", "sd11", "sd12", "sd13", "sd14", "sd15", "sd16", "sd17",
           "sd18", "sd19", "sd20", "sd21", "sd22", "sd23", "sd24", "sd25", "sd26", "sd27", "sd28", "sd29", "sd30",
           "sa_max", "sa_avg", "sv_max", "sv_avg", "sd_max", "sd_avg", "pga", "pgv", "pgd", "epa", "epv", "epd", "pa",
           "pv", "pd", "ic"]
y_label = ["ok"]
X = df.loc[:, x_label].values
y = df.loc[:, y_label].values
ss = StandardScaler().fit(X)
new_X = ss.transform(X)

## 随机森林
rfc = RandomForestClassifier()
rfc.fit(new_X, y)

# 回归问题
## 特征过程
x_label = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9",
           "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "sa1", "sa2", "sa3", "sa4", "sa5", "sa6", "sa7", "sa8",
           "sa9", "sa10", "sa11", "sa12", "sa13", "sa14", "sa15", "sa16", "sa17", "sa18", "sa19", "sa20", "sa21",
           "sa22", "sa23", "sa24", "sa25", "sa26", "sa27", "sa28", "sa29", "sa30", "sv1", "sv2", "sv3", "sv4", "sv5",
           "sv6", "sv7", "sv8", "sv9", "sv10", "sv11", "sv12", "sv13", "sv14", "sv15", "sv16", "sv17", "sv18", "sv19",
           "sv20", "sv21", "sv22", "sv23", "sv24", "sv25", "sv26", "sv27", "sv28", "sv29", "sv30", "sd1", "sd2", "sd3",
           "sd4", "sd5", "sd6", "sd7", "sd8", "sd9", "sd10", "sd11", "sd12", "sd13", "sd14", "sd15", "sd16", "sd17",
           "sd18", "sd19", "sd20", "sd21", "sd22", "sd23", "sd24", "sd25", "sd26", "sd27", "sd28", "sd29", "sd30",
           "sa_max", "sa_avg", "sv_max", "sv_avg", "sd_max", "sd_avg", "pga", "pgv", "pgd", "epa", "epv", "epd", "pa",
           "pv", "pd", "ic"]
y_label = ["drift_max", "drift_avg", "a_max", "a_avg"]
X = df.loc[:, x_label].values
y = df.loc[:, y_label].values
df_ok_origin = df[df.ok == True]
df_ok = df_ok_origin.copy()

y_train_theta_max = y[:, 0]
y_train_theta_avg = y[:, 1]
y_train_a_max = y[:, 2]
y_train_a_avg = y[:, 3]


from sklearn.preprocessing import StandardScaler

ss = StandardScaler().fit(X)
new_X = ss.transform(X)

GBoost_theta_max = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                             max_depth=4, max_features='sqrt',
                                             min_samples_leaf=15, min_samples_split=10,
                                             loss='huber', random_state=5)
GBoost_theta_avg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                             max_depth=4, max_features='sqrt',
                                             min_samples_leaf=15, min_samples_split=10,
                                             loss='huber', random_state=5)
GBoost_a_max = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                         max_depth=4, max_features='sqrt',
                                         min_samples_leaf=15, min_samples_split=10,
                                         loss='huber', random_state=5)
GBoost_a_avg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                         max_depth=4, max_features='sqrt',
                                         min_samples_leaf=15, min_samples_split=10,
                                         loss='huber', random_state=5)




select_model_theta_max = GBoost_theta_max
select_model_theta_avg = GBoost_theta_avg
select_model_a_max = GBoost_a_max
select_model_a_avg = GBoost_a_avg

select_model_theta_max.fit(new_X, y_train_theta_max)
select_model_theta_avg.fit(new_X, y_train_theta_avg)
select_model_a_max.fit(new_X, y_train_a_max)
select_model_a_avg.fit(new_X, y_train_a_avg)

joblib.dump(rfc, "./ml_models/rfc.model")
joblib.dump(GBoost_a_avg, "./ml_models/GBoost_a_avg.model")
joblib.dump(GBoost_a_max, "./ml_models/GBoost_a_max.model")
joblib.dump(GBoost_theta_avg, "./ml_models/GBoost_theta_avg.model")
joblib.dump(GBoost_theta_max, "./ml_models/GBoost_theta_max.model")

# train_pred_theta_max = select_model_theta_max.predict(new_X_train)
# pred_theta_max = select_model_theta_max.predict(new_X_test)
# print(rmsle(y_train_theta_max, train_pred_theta_max))
#
# train_pred_theta_avg = select_model_theta_avg.predict(new_X_train)
# pred_theta_avg = select_model_theta_avg.predict(new_X_test)
# print(rmsle(y_train_theta_avg, train_pred_theta_avg))
#
# train_pred_a_max = select_model_a_max.predict(new_X_train)
# pred_a_max = select_model_a_max.predict(new_X_test)
# print(rmsle(y_train_a_max, train_pred_a_max))
#
# train_pred_a_avg = select_model_a_avg.predict(new_X_train)
# pred_a_avg = select_model_a_avg.predict(new_X_test)
# print(rmsle(y_train_a_avg, train_pred_a_avg))
