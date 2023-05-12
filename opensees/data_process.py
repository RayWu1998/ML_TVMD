W# 用来处理opensees计算出的数据

import os
import pandas as pd


def combine_csv(path):
    """
    将多个设备计算出的csv文件进行合并
    :param path: cs文件的存放路径
    :return: 返回合并后的DataFrame格式数据
    """
    files = os.listdir(path)
    design_res = pd.read_csv(path + "/" + files[0])

    for file in files[1: -1]:
        df = pd.read_csv(path + "/" + file)
        design_res = pd.concat([design_res, df], axis=0, ignore_index=True)
    design_res = design_res.loc[design_res['T_1'] < 30]
    design_res = design_res.drop_duplicates()  # 去重

    design_res = design_res.reset_index(drop=True)  # 重新生成index
    return design_res


def merge_with_wave_features(opensees_design_res, wave_features_file):
    """
    和wave_features表进行内连接
    :param opensees_design_res: opensees的设计结果
    :param wave_features_file: wave_features存放的csv文件的路径
    :return: 返回内连接后的DataFrame格式数据
    """
    wave_features = pd.read_csv(wave_features_file)
    merge_res = pd.merge(left=opensees_design_res, right=wave_features, on="wave_name", how="inner")
    merge_res = merge_res.drop_duplicates()
    return merge_res


if __name__ == "__main__":
    design_res = combine_csv("./res")
    merge_res = merge_with_wave_features(design_res, "./wave_features.csv")
    # 存储为csv文件
    merge_res.to_csv("./data/opensees_design_res.csv", index=False)