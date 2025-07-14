import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
from astropy.io import fits as pyfits
from datetime import datetime, timedelta
# matplotlib.use('Qt5Agg')
from astropy.stats import sigma_clipped_stats
from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style


def main():
    # **读取参数配置文件**
    config_filenm = sys.argv[1]
    with open(config_filenm, 'r') as f:
        config = yaml.safe_load(f)
    # target_ra, target_dec = config['target_radec']  # 读取目标天球坐标 (RA, Dec)
    target_nm = config['target_nm']  # 目标名称
    raper_chos = config['raper_chos']
    r_cho, r_in, r_out, r_step, r_all = raper_chos
    # print(r_cho, r_in, r_out, r_step, r_all)
    t0 = pd.to_datetime(config['t0'])
    # print(t0)

    out_dir = 'res'
    r_cho_text = f"{r_cho:.1f}".replace('.', '_')
    lc_csvnm = f"{target_nm}_lc_{r_cho_text}.csv"
    origin_csv = os.path.join(out_dir, lc_csvnm)
    res_csv = os.path.join(out_dir, f"{target_nm}_VTtest.csv")

    # 整理结果
    # 读取origin_csv，删除 sn, r_all, mag, mag_err, cmag, cmag_err, magc_limit_err 列
    df = pd.read_csv(origin_csv)
    drop_cols = ['sn', 'r_all', 'mag', 'mag_err', 'cmag', 'cmag_err', 'magc_limit_err']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    # 比较每行的 mag_c 和 magc_limit
    # 如果 mag_c < magc_limit，则把 mag_c 的值改为 magc_limit，把 mag_c_err 的值改为 'mag_limit'
    def process_row(row):
        try:
            mag_c = float(row['mag_c'])
            magc_limit = float(row['magc_limit'])
        except Exception:
            return row
        if mag_c > magc_limit:
            row['mag_c'] = magc_limit
            row['mag_c_err'] = 'mag_limit'
        return row
    df = df.apply(process_row, axis=1)

    # mag_c, magc_limit 保留两位小数（如果不是'mag_limit'等字符串）
    def format_float(val):
        try:
            return f"{float(val):.2f}"
        except Exception:
            return val
    df['mag_c'] = df['mag_c'].apply(format_float)
    df['magc_limit'] = df['magc_limit'].apply(format_float)
    # mag_c_err 只保留两位小数（如果不是字符串）
    def format_err(val):
        if isinstance(val, str) and val == 'mag_limit':
            return val
        try:
            return f"{float(val):.2f}"
        except Exception:
            return val
    df['mag_c_err'] = df['mag_c_err'].apply(format_err)

    # 重命名列
    rename_dict = {'mag_c': 'mag', 'mag_c_err': 'mag_err', 'magc_limit': 'mag_limit', 'expt': 'exptime'}
    df = df.rename(columns=rename_dict)

    # 保存结果
    df.to_csv(res_csv, index=False)


if __name__ == '__main__':
    start_time = time.time()

    main()

    # Cal Time
    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')
