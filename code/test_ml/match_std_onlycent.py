#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import yaml
import pyds9
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.wcs import WCS
from astropy.io import fits as pyfits
from astropy.wcs import FITSFixedWarning
from scipy.spatial import KDTree, cKDTree
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.simplefilter('ignore', category=FITSFixedWarning)

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style

import findstars  # 已有的画图模块


def plot_target(img, pos_csvnm, target_id, id_colnm='obj_id'):
    '''
    使用 DS9 圈出所找到的星
    '''
    hdu = pyfits.open(img)
    header = hdu[0].header
    naxis1, naxis2 = int(header['NAXIS1']), int(header['NAXIS2'])
    d = pyds9.DS9()  # will open a new ds9 window or connect to an existing one
    # d.set(f'width {naxis1}')
    # d.set(f'height {naxis2}')
    d.set('width 1200')
    d.set('height 1200')
    d.set(f'file {img}')  # send the file to the open ds9 session
    d.set('scale zscale')  # 设置 scale 为 zscale
    d.set('zoom to 4')  # 适应窗口大小
    r_aper = 3
    df = pd.read_csv(pos_csvnm)
    for index, row in df.iterrows():  # 遍历筛选后的数据并打印 regions add circle 命令
        nid = row[id_colnm]
        nxi = row['xcent']
        nyi = row['ycent']
        if nid == target_id:  # 如果是目标星
            reg = f'image;circle({nxi},{nyi},{r_aper + 1}) # width=2 color=red text={{{int(nid)}}}'  # example region
            d.set(f'pan to {nxi} {nyi} image')  # 设置DS9中心位置
            # print(f'pan to {nxi} {nyi} image')
        else:
            reg = f'image;circle({nxi},{nyi},{r_aper}) # text={{{int(nid)}}}'  # example region
        d.set('regions', reg)
    pos_pngnm = os.path.splitext(os.path.abspath(pos_csvnm))[0] + '.png'

    # d.set(f'export png {png_nm}')  # 以高质量的方式导出 DS9 显示的内容 （只能导出一幅图）
    # print(pos_csvnm)
    # print(pos_pngnm)
    d.set(f'saveimage png {pos_pngnm}')


def find_center_star_id(pos_csvnm, header):
    '''
    找到距离视场中心最近的星，返回其id
    '''
    df = pd.read_csv(pos_csvnm)
    if 'xcent' not in df.columns or 'ycent' not in df.columns or 'obj_id' not in df.columns:
        raise ValueError(f"CSV file {pos_csvnm} is missing required columns")
    naxis1, naxis2 = int(header['NAXIS1']), int(header['NAXIS2'])
    center = np.array([naxis1 / 2, naxis2 / 2])
    coords = df[['xcent', 'ycent']].values
    dists = np.linalg.norm(coords - center, axis=1)
    idx = np.argmin(dists)
    star_id = df.iloc[idx]['obj_id']
    return star_id


def main():
    start_time = time.time()

    # **读取参数配置文件**
    config_filenm = sys.argv[1]
    with open(config_filenm, 'r') as f:
        config = yaml.safe_load(f)
    target_ra, target_dec = config['target_radec']  # 读取目标天球坐标 (RA, Dec)
    target_nm = config['target_nm']  # 目标名称
    # raper_chos = config['raper_chos']
    # r_cho, r_in, r_out, r_step, r_all = raper_chos
    # print(r_cho, r_in, r_out, r_step, r_all)
    # t0 = pd.to_datetime(config['t0'])
    # print(t0)

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

    cent_aphot_list = []
    err_lst = []  # 记录错误文件

    for k in tqdm(range(len(imgs)), desc='Step 4. Center Star Photometry'):
        try:
            hdu = pyfits.open(imgs[k])
            header = hdu[0].header
            pos_csvnm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_pos.csv'
            parq_nm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_aphot.parquet'
            if not os.path.exists(pos_csvnm) or not os.path.exists(parq_nm):
                raise FileNotFoundError(f"{pos_csvnm} or {parq_nm} not found")
            # 找到中心星id
            center_id = find_center_star_id(pos_csvnm, header)
            # 读取测光表
            df_aphot = pd.read_parquet(parq_nm)
            # 只保留该id的所有孔径测光结果
            df_center = df_aphot[df_aphot['obj_id'] == center_id].copy()
            # 增加文件名信息便于追踪
            df_center['img'] = os.path.basename(imgs[k])
            cent_aphot_list.append(df_center)
            # 画出星星位置
            plot_target(imgs[k], pos_csvnm, center_id)
        except Exception as e:
            err_lst.append(imgs[k])
            print(f'Error processing {imgs[k]}: {e}')

    # 汇总保存
    if cent_aphot_list:
        df_cent_all = pd.concat(cent_aphot_list, ignore_index=True)
        df_cent_all.to_csv('cent_aphot.csv', index=False)
        print(f'cent_aphot.csv saved, {len(df_cent_all)} rows.')
    else:
        print('No center star photometry data found.')


    # **记录错误文件以及成功处理的文件**
    if err_lst:
        with open('err4_match.lst', 'a') as err_file:
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
