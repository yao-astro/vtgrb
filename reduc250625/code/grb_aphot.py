#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sun Jul 6 20:36 2025

@author: Zhuheng_Yao
"""

import os
import ast
import sys
import time
import glob
import json
import yaml
import traceback
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits as pyfits
from datetime import datetime, timedelta
from matplotlib.pyplot import MultipleLocator
# matplotlib.use('Qt5Agg')
from astropy.stats import sigma_clipped_stats
from concurrent.futures import ThreadPoolExecutor
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style

import aphot


def main():
    start_time = time.time()

    # **读取参数配置文件**
    config_filenm = sys.argv[1]
    with open(config_filenm, 'r') as f:
        config = yaml.safe_load(f)
    # target_ra, target_dec = config['target_radec']  # 读取目标天球坐标 (RA, Dec)
    target_nm = config['target_nm']  # 目标名称
    raper_chos = config['raper_chos']
    r_cho, r_in, r_out, r_step, r_all = raper_chos
    # print(r_cho, r_in, r_out, r_step, r_all)
    # t0 = pd.to_datetime(config['t0'])
    # print(t0)
    # rlst = np.round([r_cho, r_all], 1).tolist()  # 孔径测光半径 list
    # rlst = np.round(np.arange(r_step, r_in + r_step, r_step), 1).tolist()  # 计算孔径半径列表
    # print(rlst)
    
    # 读取grb_xyposs.csv，获取每帧目标源的xy位置
    grb_xy_df = pd.read_csv('grb_xyposs.csv')
    imgs = grb_xy_df['frame'].tolist()
    grb_xys = dict(zip(grb_xy_df['frame'], zip(grb_xy_df['xcent'], grb_xy_df['ycent'])))
    print(f'Total {len(imgs)} frames, target pixel positions loaded from grb_xyposs.csv')

    ###########################################################################
    #                           Aperture Photometry                           #
    ###########################################################################
    err_lst = []  # List to store images that cause errors
    all_img_results = []  # 用于收集所有帧的测光结果
    for k in tqdm(range(len(imgs)), desc='Step 5. Aperture Photometry'):
        try:
            frame = imgs[k]
            df_img = pd.DataFrame()  # 创建一个空的 DataFrame 来存储所有的表格数据
            phot = aphot.photut_aper(frame, [grb_xys[frame]], r_cho, r_in, r_out)
            df_phot = phot.to_pandas()  # 将 QTable 转换为 pandas DataFrame
            df_img = pd.concat([df_img, df_phot], ignore_index=True)  # 将数据追加到主 DataFrame 中
            df_img['frame'] = frame  # 增加frame列
            all_img_results.append(df_img)
        except Exception as e:
            traceback.print_exc()
            err_lst.append(imgs[k])  # Add the current image to the error list
    # 合并所有帧的测光结果并保存
    if all_img_results:
        df_all = pd.concat(all_img_results, ignore_index=True)
        df_all.to_csv('grb_aphot.csv', index=False)
        print('All photometry results have been saved to grb_aphot.csv')

    ###########################################################################
    #                             Calculate Time                              #
    ###########################################################################
    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')


if __name__ == "__main__":
    main()
