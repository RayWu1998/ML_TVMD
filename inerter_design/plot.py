import pandas as pd
import os
import shutil
import numpy as np

# 绘图
import matplotlib.pyplot as plt


# 基本工具
import matplotlib
import platform
import numpy as np
import os
import pandas as pd


# 样本数据处理
from sklearn.model_selection import train_test_split

# 分类模型
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve

# 回归模型
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error

# 保存模型
import joblib

# 贝叶斯优化
from skopt import BayesSearchCV


sys_platform = platform.platform().lower()
if "windows" in sys_platform:
    font = {
        "family": "Times New Roman"
    }
    matplotlib.rc("font", **font)
else:
    font = {
        "family": "Arial Unicode MS"
    }
    matplotlib.rc("font", **font)
rc = {"mathtext.fontset": "stix", }

plt.rcParams.update(rc)

## 图像显示中文的问题，需要判断系统是windows还是苹果的

sys_platform = platform.platform().lower()
if "windows" in sys_platform:
    font = {
        "family": "Microsoft YaHei"
    }
    matplotlib.rc("font", **font)
else:
    font = {
        "family": "Arial Unicode MS"
    }
    matplotlib.rc("font", **font)


def combine_csv(path):
    """
    将多个设备计算出的csv文件进行合并
    :param path: cs文件的存放路径
    :return: 返回合并后的DataFrame格式数据
    """
    files = os.listdir(path)
    design_res = pd.read_csv(path + "/" + files[0], )

    for file in files[1:]:
        df = pd.read_csv(path + "/" + file)
        design_res = pd.concat([design_res, df], axis=0)
    design_res = design_res.drop_duplicates()  # 去重
    design_res = design_res.reset_index(drop=True)  # 重新生成index
    design_res = design_res.loc[:, ~design_res.columns.str.contains('Unnamed')]
    return design_res

data_dir = "./data"
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir, 0o777)
design_res = combine_csv("./design_res_final_beta")
tar_file = os.path.join(data_dir, "design_res_beta.csv")
# 存储为csv文件
design_res.to_csv(tar_file, index=False)

df=pd.read_csv(tar_file)
df.info()
df.sample(3)

df = df.dropna(axis=0, how='any')

x_label = ["period", "site_no", "wg", "xig", "stru_damping", "gamma_t"]
y_label = ["miu", "zeta", "kappa"]
df_ok_origin = df[df.ok==True]
# df_ok_origin = df
df_ok = df_ok_origin.copy()

X = df_ok.loc[:, x_label].values
y = df_ok.loc[:, y_label].values

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=10, azim=160)
ax.scatter(y[:,0], y[:,1], y[:,2], alpha=0.2)
ax.set_xlabel(r'$\mu$', fontsize=15)
ax.set_ylabel(r'$\xi$', fontsize=15)
ax.set_zlabel(r'$\kappa$', fontsize=15)
pic_name = r"设计参数分布视角2"
plt.savefig(r'./pic/{}.png'.format(pic_name), dpi=600, bbox_inches='tight')
plt.show()
plt.close()
