#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ast
import sys
import glob
import time
import yaml
import warnings
import traceback
import numpy as np
import pandas as pd
import scipy.odr as odr
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)  # 忽略所有 DtypeWarning 警告

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style


def main():
    start_time = time.time()
    set_mpl_style()
    markersize, elw = 10, 0.5  # 误差棒的线宽

    grb_aphot_path = 'grb_aphot.csv'
    df_grb = pd.read_csv(grb_aphot_path)


    # **检查有无 magc 目录**
    magc_dir = 'magc'
    if not os.path.exists(magc_dir):  # 如果不存在 magc 目录则不再继续
        print(f'No [{magc_dir}] Dir, back to Step x. correcting Magnitudes')
        sys.exit(1)

    rows = []
    for idx, row in df_grb.iterrows():
        frame_path = row['frame']
        # 2. 提取fits名主体
        fits_name = os.path.splitext(os.path.basename(frame_path))[0]
        magc_csv = os.path.join(magc_dir, f'{fits_name}_magc.csv')
        if not os.path.exists(magc_csv):
            continue
        df_magc = pd.read_csv(magc_csv)
        # 3. 匹配r_aper==r1
        r_aper = row['r_aper']
        matched = df_magc[df_magc['r1'] == r_aper]
        for _, magc_row in matched.iterrows():
            new_row = row.copy()
            new_row['mag_c'] = row['mag'] + magc_row['c']
            new_row['mag_c_err'] = np.sqrt(row['mag_err']**2 + magc_row['c_err']**2)
            new_row['r_all'] = magc_row['r2']
            new_row['cmag'] = magc_row['c']
            new_row['cmag_err'] = magc_row['c_err']
            rows.append(new_row)

    # 4. 合并所有新行，保存，只保留指定列
    df_new = pd.DataFrame(rows)
    keep_cols = ['t_start', 'mag_c', 'mag_c_err', 'band', 'expt', 'ncombine', 
                 'sn', 'r_aper', 'r_all', 'mag', 'mag_err', 'cmag', 'cmag_err']
    df_new = df_new[keep_cols]
    df_new.to_csv('grb_aphot_magc.csv', index=False)

    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.3f} s')


if __name__ == "__main__":
    main()
