import pandas as pd

data = {"I0": [31.42, 25.13, 20.94],
        "I1": [25.13, 20.94, 17.95],
        "II": [17.95, 15.71, 13.96],
        "III": [13.96, 11.42, 9.67],
        "IV": [9.67, 8.38, 6.98]
        }
wg = pd.DataFrame(data, index=["设计地震第1组", "设计地震第2组", "设计地震第3组"])
site_cate = wg.columns
group = wg.index
xig = pd.Series([0.64, 0.64, 0.72, 0.80, 0.90], index=["I0", "I1", "II", "III", "IV"])
site_cond = pd.DataFrame(columns=["类型编号", "类型名称", "wg", "xig"])
wg.to_csv("./res.csv", encoding="gbk")
