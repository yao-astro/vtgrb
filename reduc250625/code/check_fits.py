#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.io import fits as pyfits

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style


def main():
    start_time = time.time()
    set_mpl_style()
    markersize, elw = 10, 0.5  # 误差棒的线宽

    # **检查 fit 列表**
    # fit_lstnm = sys.argv[1]
    # fit_lst = np.loadtxt(fit_lstnm, dtype=str)
    # fit_lst = np.atleast_1d(fit_lst).tolist()
    # fit_lst.sort()
    # imgs = [i.strip() for i in fit_lst]
    # if not imgs:  # 检查 imgs 是否为空（如果不为空则继续执行，如果为空则不再执行）
    #     print(f'Error: {fit_lstnm} is empty, Exiting <{os.path.basename(__file__)}>...')
    #     sys.exit(1)  # 退出程序，返回状态码 1
    
    # **检查并读取 fit list**
    fit_lstnm = sys.argv[1]
    if not os.path.exists(fit_lstnm):  # 检查文件是否存在
        print(f'Error [{os.path.basename(__file__)}]: {fit_lstnm} does not exist, Exit...')
        sys.exit(1)
    if os.stat(fit_lstnm).st_size == 0:  # 检查文件是否为空
        print(f'Error [{os.path.basename(__file__)}]: {fit_lstnm} is empty, Exit...')
        sys.exit(1)
    fit_lst = np.loadtxt(fit_lstnm, dtype=str)
    fit_lst = np.atleast_1d(fit_lst).tolist()
    fit_lst.sort()
    imgs = [i.strip() for i in fit_lst]
    if not imgs:  # 检查 imgs 是否为空（如果不为空则继续执行，如果为空则不再执行）
        print(f'Error [{os.path.basename(__file__)}]: {fit_lstnm} is empty, Exit...')
        sys.exit(1)

    # **检查 fit 文件是否缺少关键字以及 WCS 信息，如果缺少的话就写入 err_lst**
    required_keys = ['DATE-OBS', 'BAND', 'BGMEDIAN', 'BGSTD', 'READRATE', 'GAIN', 'EXPTIME',
                     'NAXIS1', 'NAXIS2']  # 需要检查的 fit 头关键字
                    # , 'STAB_CNT', 'EE70', 'EE70_RMS', 'EE80', 'EE80_RMS']
    err_lst = []
    for k in tqdm(range(len(imgs)), desc='Step 1. Checking Fits'):
        try:
            hdu = pyfits.open(imgs[k])
            header = hdu[0].header
            # **检查关键字是否齐全**
            missing_keys = [key for key in required_keys if key not in header]
            if missing_keys:  # 如果缺少关键字，记录到错误列表
                err_lst.append(imgs[k])
                continue  # 跳过后续检查，直接进入下一个文件
            # 检查READRATE参数是否为200kHz
            readrate = header.get('READRATE', None)
            if readrate is None or str(readrate).lower() != '200khz':
                err_lst.append(imgs[k])
                continue
            # 检查BGMEDIAN范围
            bgmedian = header.get('BGMEDIAN', None)
            try:
                bgmedian_val = float(bgmedian)
            except Exception:
                bgmedian_val = None
            if bgmedian_val is None or not (0 < bgmedian_val < 800):
                err_lst.append(imgs[k])
                continue
            # 检查图像尺寸是否合理 naxis1必须等于naxis2
            naxis1 = header.get('NAXIS1', None)
            naxis2 = header.get('NAXIS2', None)
            try:
                naxis1_val = int(naxis1)
                naxis2_val = int(naxis2)
            except Exception:
                naxis1_val = naxis2_val = None
            if naxis1_val is None or naxis2_val is None or naxis1_val != naxis2_val:
                err_lst.append(imgs[k])
                continue
            # 检查图像稳定度 stab_cnt/exptime 要大于50%
            stab_cnt = header.get('STAB_CNT', None)
            exptime = header.get('EXPTIME', None)
            if stab_cnt is not None:
                try:
                    stab_cnt_val = float(stab_cnt)
                    exptime_val = float(exptime)
                except Exception:
                    stab_cnt_val = exptime_val = None
                if stab_cnt_val is None or exptime_val is None or exptime_val == 0 or (stab_cnt_val / exptime_val) < 0.5:
                    err_lst.append(imgs[k])
                    continue
            # GRB必须要有WCS信息
            # **检查 WCS 信息**
            wcs = WCS(header)
            has_wcs = wcs.is_celestial
            if missing_keys or not has_wcs:  # 如果缺少关键字或 WCS 错误，记录到错误列表
                err_lst.append(imgs[k])
        except Exception as e:
            err_lst.append(imgs[k])

    # **记录错误文件以及成功处理的文件**
    if err_lst:
        with open('err1_check.lst', 'a') as err_file:
            for img in err_lst:
                err_file.write(f'{img}\n')
    suc_lst = [img for img in imgs if img not in err_lst]
    with open('suc.lst', 'w') as suc_file:
        for img in suc_lst:
            suc_file.write(f'{img}\n')

    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')


if __name__ == "__main__":
    main()
