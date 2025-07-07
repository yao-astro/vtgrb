#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 13 16:21 2024

v1: Edited in Fri Nov 29 16:41 2024
v2: Edited in Wed Dec 25 21:05 2024

@author: Zhuheng_Yao
"""

import os
import re
import ast
import sys
import time
import glob
import math
import yaml
import shutil
import colorsys
import numpy as np
import pandas as pd
import matplotlib as mpl
import plotly.subplots as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from astropy.io import fits as pyfits
from datetime import datetime, timedelta
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.use('Qt5Agg')
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style


import grb_lcplot


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
    t0 = pd.to_datetime(config['t0'])  # 读取 T0 时间

    # 剔除离群点和大误差点
    # df_sel = filter_outliers(df_sel, mag_col='mag', mag_err_col='mag_err', sigma=3, max_mag_err=0.2, max_iter=5)
    r_cho_text = f'{r_cho:.1f}'.replace('.', '_')
    lc_csvnm = f'{target_nm}_lc_{r_cho_text}.csv'
    lc_csv = os.path.join('res', lc_csvnm)
    if not os.path.exists(lc_csv):
        print(f'Error: {lc_csv} does not exist, Exit...')
        sys.exit(1)
        
    # 读取光变曲线数据
    df = pd.read_csv(lc_csv)
    grb_lcplot.plot_lc_html(lc_csv, target_nm, t0)

    # Cal Time
    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')


if __name__ == "__main__":
    main()
