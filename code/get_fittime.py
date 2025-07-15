#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as U
from astropy.io import fits as pyfits
from astropy import constants as const

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style


def get_fitpath(fitnm, data_path='/home/vxpp/program/data/L2'):
    """
    直接在 data_path 下递归查找所有子目录，返回第一个匹配的 FITS 文件路径
    """
    fitpaths = glob.glob(os.path.join(data_path, '**', fitnm), recursive=True)
    if fitpaths:
        # print(fitpaths[0])
        return fitpaths[0]
    else:
        return None


def get_endtime(t_start, exptime, ncombine=1):
    '''
    直接根据 ncombine 计算结束时间
    '''
    t_end = pd.to_datetime(t_start) + pd.Timedelta(seconds=exptime * ncombine)
    # 返回ISO格式字符串（带微秒）
    t_end_str = t_end.isoformat()
    # print(t_start, exptime, ncombine, t_end_str)
    return t_end_str


def get_time(fit, data_path='/home/vxpp/program/data/L2'):
    '''
    获取FITS文件的观测起止时间（t_start, t_end）
    参数：
        fit : str
            FITS文件路径
    返回：
        t_start : str or None
            观测起始时间（DATE-OBS，字符串）
        t_end : pandas.Timestamp or None
            观测结束时间（自动根据EXPTIME和NCOMBINE推算，或合并帧最后一张的结束时间）
    说明：
        - 若ncombine=1，t_end = t_start + exptime
        - 若ncombine>1，优先查找头部IMCMBxxx字段指定的最后一帧，取其t_start+exptime为t_end
        - 若找不到最后一帧，则用t_start + exptime * ncombine 近似
        - 若fit文件不存在，返回(None, None)
    '''
    if fit is None or not os.path.exists(fit):
        print(f'Error: FITS file not found: {fit}')
        return None, None
    with pyfits.open(fit) as hdul:
        ncombine = hdul[0].header.get('NCOMBINE', 1)
        t_start = hdul[0].header.get('DATE-OBS')
        exptime = hdul[0].header.get('EXPTIME')
    if ncombine == 1:  # 如果 ncombine == 1，直接计算结束时间
        t_end = get_endtime(t_start, exptime)
    else:  # 如果 ncombine > 1，通过参与合并的最后一帧来计算结束时间
        data_path = os.path.dirname(fit).replace('imastk_vt', 'imacal_vt')
        # print(data_path)  # 获取更准确的数据路径
        end_fitnm = hdul[0].header.get(f'IMCMB{int(ncombine):03d}', 'unknown')
        end_fitnm = end_fitnm.replace('_2sd.fit', '.fit')  # 替换为原始 FITS 文件名
        end_fitpath = get_fitpath(end_fitnm, data_path=data_path)  # 获取参与合并的最后一帧fit文件路径
        if end_fitpath:
            # print(1)
            with pyfits.open(end_fitpath) as end_hdul:
                end_tstart = end_hdul[0].header.get('DATE-OBS')
                end_exptime = end_hdul[0].header.get('EXPTIME')
                t_end = get_endtime(end_tstart, end_exptime)
        else:
            # print(2)
            t_end = get_endtime(t_start, exptime, ncombine)
    return t_start, t_end

def main():
    start_time = time.time()
    set_mpl_style()
    markersize, elw = 10, 0.5  # 误差棒的线宽

    # # 给定 SVT的fits 文件名
    # fitnm = 'SVT_ToO-NOM-GRB_02842_R_02185_241116T192856_46130.fit'
    # fitpath = get_fitpath(fitnm)
    # print(fitpath)

    fitnm = 'SVT_ToO-EX_02786_B_02141_241113T200956_40551_c_n254.fit'
    if os.path.exists(fitnm):
        fitpath = fitnm
    else:
        fitpath = get_fitpath(fitnm)
    t_start, t_end = get_time(fitpath)
    print(t_start, t_end)

    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')


if __name__ == "__main__":
    main()
