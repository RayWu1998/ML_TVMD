import numpy as np
import pandas as pd
from cal_wave_features import get_all_features
import re

if __name__ == "__main__":
    miu = np.random.random(9)
    xi = np.random.random(9)
    kappa = np.random.random(9)
    md = np.zeros(9)
    cd = np.zeros(9)
    kd = np.zeros(9)
    # 惯容系统参数*
    # 1
    md[0] = miu[0] * 504675.54
    cd[0] = 2 * np.power(504675.54 * 17701517.6367, 0.5) * xi[0]
    kd[0] = 17701517.6367 * kappa[0]
    # 2
    md[1] = miu[1] * 494164.64
    cd[1] = 2 * np.power(494164.64 * 212966222.7839, 0.5) * xi[1]
    kd[1] = 212966222.7839 * kappa[1]
    # 3
    md[2] = miu[2] * 494164.64
    cd[2] = 2 * np.power(494164.64 * 198709990.1556, 0.5) * xi[2]
    kd[2] = 198709990.1556 * kappa[2]
    # 4
    md[3] = miu[3] * 494164.64
    cd[3] = 2 * np.power(494164.64 * 178923770.7303, 0.5) * xi[3]
    kd[3] = 178923770.7303 * kappa[3]
    # 5
    md[4] = miu[4] * 494164.64
    cd[4] = 2 * np.power(494164.64 * 168236912.4272, 0.5) * xi[4]
    kd[4] = 168236912.4272 * kappa[4]
    # 6
    md[5] = miu[5] * 494164.64
    cd[5] = 2 * np.power(494164.64 * 153412925.7595, 0.5) * xi[5]
    kd[5] = 153412925.7595 * kappa[5]
    # 7
    md[6] = miu[6] * 494164.64
    cd[6] = 2 * np.power(494164.64 * 117202366.5283, 0.5) * xi[6]
    kd[6] = 117202366.5283 * kappa[6]
    # 8
    md[7] = miu[7] * 494164.64
    cd[7] = 2 * np.power(494164.64 * 84968874.1376, 0.5) * xi[7]
    kd[7] = 84968874.1376 * kappa[7]
    # 9
    md[8] = miu[8] * 534637.2
    cd[8] = 2 * np.power(534637.2 * 58205933.4045, 0.5) * xi[8]
    kd[8] = 58205933.4045 * kappa[8]

    filename = "./waves_test/" + "RSN8164_DUZCE_487-NS.AT2";
    with open(filename, 'r') as fo:
        for i in range(3):
            fo.readline()
        line = fo.readline()
        target = re.search(r'DT= *(.*) *SEC', line, re.I | re.M)
        dt = float(target.group(1))
        wave = pd.read_table(filename, skiprows=4, sep="  ", header=None, engine='python')
        wave = wave.fillna(0)
        wave = wave.values
        size = wave.size
        wave = wave.reshape((size,))
        wave = wave * 9.8
    xi = 0.02
    [SA, SV, SD, SA_avg, SV_avg, SD_avg, SA_max, SV_max, SD_max, PGA, PGV, PGD, EPA, EPV, EPD, Pa, Pv, Pd,
     Ic] = get_all_features(wave, dt, xi)
    input_params = pd.DataFrame(data=None,
                                columns=["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "k1", "k2", "k3", "k4",
                                         "k5", "k6", "k7", "k8", "k9", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
                                         "c9", "sa1", "sa2", "sa3", "sa4", "sa5", "sa6", "sa7", "sa8", "sa9", "sa10",
                                         "sa11", "sa12", "sa13", "sa14", "sa15", "sa16", "sa17", "sa18", "sa19", "sa20",
                                         "sa21", "sa22", "sa23", "sa24", "sa25", "sa26", "sa27", "sa28", "sa29", "sa30",
                                         "sv1", "sv2", "sv3", "sv4", "sv5", "sv6", "sv7", "sv8", "sv9", "sv10", "sv11",
                                         "sv12", "sv13", "sv14", "sv15", "sv16", "sv17", "sv18", "sv19", "sv20", "sv21",
                                         "sv22", "sv23", "sv24", "sv25", "sv26", "sv27", "sv28", "sv29", "sv30", "sd1",
                                         "sd2", "sd3", "sd4", "sd5", "sd6", "sd7", "sd8", "sd9", "sd10", "sd11", "sd12",
                                         "sd13", "sd14", "sd15", "sd16", "sd17", "sd18", "sd19", "sd20", "sd21", "sd22",
                                         "sd23", "sd24", "sd25", "sd26", "sd27", "sd28", "sd29", "sd30", "sa_max",
                                         "sa_avg", "sv_max", "sv_avg", "sd_max", "sd_avg", "pga", "pgv", "pgd", "epa",
                                         "epv", "epd", "pa", "pv", "pd", "ic"])
    res = []
    for i in md:
        res.append(i)
    for i in kd:
        res.append(i)
    for i in cd:
        res.append(i)
    for i in SA:
        res.append(i)
    for i in SV:
        res.append(i)
    for i in SD:
        res.append(i)
    res.append(SA_max)
    res.append(SA_avg)
    res.append(SV_max)
    res.append(SV_avg)
    res.append(SD_max)
    res.append(SD_avg)
    res.append(PGA)
    res.append(PGV)
    res.append(PGD)
    res.append(EPA)
    res.append(EPV)
    res.append(EPD)
    res.append(Pa)
    res.append(Pv)
    res.append(Pd)
    res.append(Ic)
    input_params.loc[len(input_params)] = res