#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 13 13:44 2024

v1: Edited in Wed Dec 25 21:06 2024

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


def strip_str(str_input):
    '''
    去除字符串末尾的KHz（无论大小写），并将剩余部分转为float格式
    '''
    s_cleaned = str_input.rstrip('khzKHZ').strip()
    return float(s_cleaned)


def rd_poscsv(pos_csvnm):
    '''
    始终读取 xcent/ycent，如果有 ra、dec、obj_idm 这些列则一并读取，否则只返回已有的列，兼容不同格式的星表
    '''
    df = pd.read_csv(pos_csvnm)
    cols = ['xcent', 'ycent']
    for c in ['ra', 'dec', 'obj_idm']:
        if c in df.columns:
            cols.append(c)
    pos = df[cols].values
    return pos


def rdr_aper(aperfilnm):
    '''
    read measure apertures, r, r_in, r_out
    aper file name: 'aper.txt'
    format: r, r_in, r_out
    '''
    aperlst = open(aperfilnm).readline().split()
    aperlst = [float(i) for i in aperlst]
    return tuple(aperlst)


def photut_err(flux, gain, area, stdev, nsky, ncombine=1):
    '''
    根据误差传递求flux的误差
    err1和2是孔径内总计数，err3是背景的误差
    '''
    err1 = np.true_divide(flux, gain * ncombine)
    err2 = area * np.square(stdev)
    err3 = np.square(area) * np.true_divide(np.square(stdev), nsky)
    flux_err = np.sqrt(err1 + err2 + err3)
    return flux_err


def photut_sn(adu, area, msky, nsky, expt, dn=0, gain=1, rdn=0, ncombine=1):
    '''
    计算信噪比 S/N
        adu  --  孔径内计数值/中值
        area --  孔径内面积/像素数
        msky --  背景计数值/中值
        nsky --  背景面积/像素数
        expt --  exptime 曝光时间
        dn   --  dark noise 暗噪声
        gain --  增益
        rdn  --  read noise 读出噪声
        ncombine --  合并的图像数（默认为1）
    '''
    signal = adu * gain
    f_area = 1 + np.true_divide(area, nsky)
    errn = area * f_area * (msky * gain + expt * dn + np.square(rdn))
    sn = np.true_divide(signal, np.sqrt(signal + errn))
    sn *= np.sqrt(ncombine)  # 信噪比需要乘以 sqrt(ncombine)
    return sn


def f2mag_vt(adu, gain, expt, band, mag0=25.):
    '''
    将 VT 测量的流量计数转为星等
    B/R 的星等零点分别为 zeromagB 和 zeromag_R
    '''
    mag = mag0 - 2.5 * np.log10(adu * gain / float(expt))
    return mag


def bkg_filter(dat):
    '''
    使用四分位法，删除背景值异常的点
    '''
    Q1 = np.percentile(dat, 25)
    Q3 = np.percentile(dat, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR  # 设定剔除阈值
    upper_bound = Q3 + 1.5 * IQR
    dat_filter = dat[(dat >= lower_bound) & (dat <= upper_bound)]  # 过滤掉异常值
    # print(Q1, Q3, IQR, lower_bound, upper_bound)
    return dat_filter


def photut_aper(img, pos, r_aper, r_in, r_out):
    '''
    针对单幅图像(img)指定位置(pos)处的星，
    进行半径(r_aper)和背景(r_in ~ r_out)的孔径测光，
    '''
    hdu = pyfits.open(img)
    dat = hdu[0].data
    header = hdu[0].header

    t_start = header['DATE-OBS']  # 观测开始时间，格式为'%Y-%m-%dT%H:%M:%S.%f'
    gain = float(header['GAIN'])
    if 'NCOMBINE' in header:
        ncomb = float(header['NCOMBINE'])
    else:
        ncomb = 1
    rdn = float(header['RON'])  # 读出噪声
    expt = float(header['EXPOSURE'])  # 实际有效曝光时间
    # # 计算观测的中间时刻 t_obs 以及时间误差 t_obs_err
    # t_obs_err = expt * ncomb * 0.5  # 时间误差的秒数
    # time_fmt = '%Y-%m-%dT%H:%M:%S.%f'
    # t_obs = datetime.strptime(t_start, time_fmt) + timedelta(seconds=t_obs_err)  # 计算观测中点时间
    # t_obs = t_obs.strftime(time_fmt)[:-3]  # 精确到毫秒

    rdrate = strip_str(header['READRATE'])  # 读出速率
    # bgmedian = header['BGMEDIAN']
    # bgstd = header['BGMEDIAN']
    mag0 = 25.0  # 默认星等零点
    band = header.get('BAND', '').strip()  # 获取波段并去除空格
    if band.upper() == 'VT_B':
        mag0 = float(header.get('ZEROMAGB', mag0))  # B 波段星等零点
    else:
        mag0 = float(header.get('ZEROMAGR', mag0))  # R 波段星等零点

    # 分离坐标和id，只有在pos中存在ra、dec、obj_idm这些列时才会赋值，否则自动跳过
    pos = np.array(pos)
    pos_xy = pos[:, :2].astype(float)  # 提取 nx, ny 作为测光坐标
    obj_ra = obj_dec = obj_ids = None
    if pos.shape[1] >= 3:
        obj_ra = pos[:, 2].astype(float)
    if pos.shape[1] >= 4:
        obj_dec = pos[:, 3].astype(float)
    if pos.shape[1] >= 5:
        obj_ids = pos[:, -1].astype(int)

    aperture = CircularAperture(pos_xy, r_aper)
    annulus_aperture = CircularAnnulus(pos_xy, r_in, r_out)
    annulus_masks = annulus_aperture.to_mask(method='center')
    
    bkg_median, bkg_stdev, bkg_nsky = [], [], []
    for mask in annulus_masks:  # 对需要测光的亮星位置作循环，循环次数等于选择的星的个数
        annulus_data = mask.multiply(dat)
        annulus_data_1d = annulus_data[mask.data > 0]
        annulus_data_1d = bkg_filter(annulus_data_1d)
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        stdev = np.std(annulus_data_1d)
        nsky = annulus_data_1d.size
        bkg_median.append(median_sigclip)
        bkg_stdev.append(stdev)
        bkg_nsky.append(nsky)
    
    bkg_median = np.array(bkg_median)  # 数组维度为循环次数*1
    bkg_stdev = np.array(bkg_stdev)
    bkg_nsky = np.array(bkg_nsky)
    phot = aperture_photometry(dat, aperture)
    phot['r_aper'] = r_aper  # 孔径半径
    # phot['s_aper'] = aperture.area  # 孔径包含的面积
    phot['r_in'] = r_in  # 内半径
    phot['r_out'] = r_out  # 外半径
    # phot['bkg_nsky'] = bkg_nsky  # 背景区域的像素数
    phot['adu_bkg_median'] = bkg_median  # 背景计数中值
    # phot['bkg_stdev'] = bkg_stdev  # 背景计数标准差
    phot['adu_bkg_aper'] = aperture.area * bkg_median  # 孔径内的总背景计数
    phot['adu'] = phot['aperture_sum'] - phot['adu_bkg_aper']  # 孔径内扣除背景的计数
    phot['sn'] = photut_sn(phot['adu'], aperture.area, bkg_median, bkg_nsky, 
                           expt, dn=0, gain=gain, rdn=rdn)
    phot['adu_err'] = phot['adu'] / phot['sn']  # 根据信噪比计算计数误差
    phot['mag'] = f2mag_vt(phot['adu'], gain, expt, band, mag0)  # 将计数转换为星等
    phot['mag_err'] = 2.5 / (np.log(10) * phot['sn'])  # 根据信噪比计算星等误差
    for col in phot.colnames:
        phot[col].info.format = '%.4f'  # for consistent table output
    # phot['mag_err'].info.format = '%.8g'
    phot['gain'] = gain  # 增益
    phot['rdn'] = rdn  # 读出噪声
    phot['expt'] = expt  # 曝光时间
    phot['ncombine'] = ncomb  # 合并的图像数
    # phot['readrate'] = rdrate
    phot['t_start'] = t_start  # 观测开始时间
    # phot['t_obs'] = t_obs  # 观测中点时间
    # phot['t_obs_err'] = t_obs_err
    phot['band'] = band
    # phot表赋值时判断
    if obj_ra is not None:
        phot['ra'] = obj_ra
    if obj_dec is not None:
        phot['dec'] = obj_dec
    if obj_ids is not None:
        phot['obj_id'] = obj_ids
    # print(phot)
    # print(type(phot))  # Qtable格式文件
    return phot


def process_aphot(img, pos, rlst, r_in, r_out):
    '''
    并行计算不同 r_aper 下的测光结果
    '''
    df_list = []
    for r in rlst:
        phot = photut_aper(img, pos, r, r_in, r_out)
        df_list.append(phot.to_pandas())
    return pd.concat(df_list, ignore_index=True)


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

    # ****计算孔径测光半径列表****
    rlst = np.round(np.arange(r_step, r_in + r_step, r_step), 1).tolist()

    # **检查并读取 fit list**
    fit_lstnm = sys.argv[2]
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
    
    ###########################################################################
    #                           Aperture Photometry                           #
    ###########################################################################
    # err_lst = []  # List to store images that cause errors
    # for k in tqdm(range(len(imgs)), desc='Step 5. Aperture Photometry'):
    #     try:
    #         pos_csvnm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_pos.csv'
    #         pos = rd_poscsv(pos_csvnm)
    #         df_img = pd.DataFrame()  # 创建一个空的 DataFrame 来存储所有的表格数据
    #         rlst = np.round(np.arange(r_step, r_all + r_step, r_step), 1).tolist()  # 设置孔径半径
    #         # print(rlst)
    #         # for j in range(1):  # test
    #         for j in range(len(rlst)):
    #             phot = photut_aper(imgs[k], pos, rlst[j], r_in, r_out, target_nm)
    #             df_phot = phot.to_pandas()  # 将 QTable 转换为 pandas DataFrame
    #             df_img = pd.concat([df_img, df_phot], ignore_index=True)  # 将数据追加到主 DataFrame 中
    #         df_csvnm = os.path.basename(imgs[k]).replace('.fit', '_aphot.csv')
    #         df_img.to_csv(df_csvnm, index=False)  # 保存为一个 CSV 文件
    #     except Exception as e:
    #         traceback.print_exc()
    #         err_lst.append(imgs[k])  # Add the current image to the error list

    err_lst = []  # List to store images that cause errors
    for k in tqdm(range(len(imgs)), desc='Step 5. Aperture Photometry'):
        try:
            pos_csvnm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_pos.csv'
            pos = rd_poscsv(pos_csvnm)  # 读取星表
            # 使用线程池并行计算
            with ThreadPoolExecutor() as executor:
                df_img = executor.submit(process_aphot, imgs[k], pos, rlst, r_in, r_out).result()
            # df_csvnm = os.path.basename(imgs[k]).replace('.fit', '_aphot.csv')
            # df_img.to_csv(df_csvnm, index=False)  # 保存为 CSV 文件
            df_parqnm = os.path.basename(imgs[k]).replace('.fit', '_aphot.parquet')  # 保存为 Parquet 文件，更节省空间
            df_img.to_parquet(df_parqnm, index=False)  # 保存为 Par 文件
        except Exception as e:
            traceback.print_exc()
            err_lst.append(imgs[k])  # Add the current image to the error list

    # **记录错误文件以及成功处理的文件**
    if err_lst:
        with open('err5_aphot.lst', 'a') as err_file:
            for img in err_lst:
                err_file.write(f'{img}\n')
    suc_lst = [img for img in imgs if img not in err_lst]
    with open('suc.lst', 'w') as suc_file:
        for img in suc_lst:
            suc_file.write(f'{img}\n')

    ###########################################################################
    #                             Calculate Time                              #
    ###########################################################################
    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')


if __name__ == "__main__":
    main()
