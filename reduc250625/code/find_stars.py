#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sep
import sys
import glob
import math
import time
# import pyds9
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.wcs import WCS
from astropy import units as u
from scipy.spatial import cKDTree
from astropy.io import fits as pyfits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.detection import IRAFStarFinder
from photutils.datasets import load_star_image
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)


def del_boundstars(header, df):
    '''
    筛选掉图像边缘的星
    '''
    naxis1, naxis2 = header['NAXIS1'], header['NAXIS2']
    nbound1, nbound2 = int(naxis1 / 10), int(naxis2 / 10)
    df = df[(df['xcent'] <= naxis1 - nbound1) & (df['xcent'] >= nbound1)]
    df = df[(df['ycent'] <= naxis2 - nbound2) & (df['ycent'] >= nbound2)]
    return df


def findstars_sep(img, pos_csvnm, nthreshold=1.5):
    '''
    寻找视场里面所有的星并保存它们的 xy 坐标信息到 pos_csvnm 中
    '''
    hdu = pyfits.open(img)
    header = hdu[0].header
    gain = float(header['GAIN'])
    # dat = hdu[0].data

    dat = hdu[0].data.astype(np.float32)  # 确保数据是 float32 类型
    dat = dat.astype(dat.dtype.newbyteorder('='))  # 转换为原生字节序，避免 sep 的错误

    bkg = sep.Background(dat)  # 背景建模
    # print(bkg.globalback)
    # print(nthreshold * bkg.globalrms)
    sep.set_extract_pixstack(1000000)  # 增大像素栈限制，防止 internal pixel buffer full
    sep.set_sub_object_limit(1000000)  # 增大子对象限制，防止 object deblending overflow
    objects = sep.extract(dat - bkg, nthreshold, err=bkg.globalrms, gain=gain)
    # print(objects)
    # print(type(objects))

    df = pd.DataFrame(objects)  # 将 objects 转换为 DataFrame
    if df.empty:
        # 保存空表，包含表头
        df = pd.DataFrame(columns=['obj_id', 'xcent', 'ycent', 'flux', 'peak', 'ra', 'dec'])
        df.to_csv(pos_csvnm, index=False)
        return
    df = df.rename(columns={'x': 'xcent', 'y': 'ycent'})
    df[['xcent', 'ycent']] += 1  # 加 1 修正坐标（加了 1 之后，xcent 和 ycent 就是 FITS 原像素坐标，坐标从 1 开始）
    df = del_boundstars(header, df)  # 删除找到的离图像边缘太近的源
    df = df[df['flux'] > 0]  # 提取出 flux 大于 0 的星
    df = df[df['peak'] < 65000]  # 提取出 peak 没有过饱和（大于65000）的星
    # 同时保存 RA Dec
    wcs = WCS(header)
    ra, dec = wcs.all_pix2world(df['xcent'], df['ycent'], 1)
    df['ra'] = ra
    df['dec'] = dec
    # 计算 ra dec 和 fit header 里面的 obj_ra obj_dec 的角距离，写为一列
    obj_ra = float(header.get('OBJ_RA', 0))
    obj_dec = float(header.get('OBJ_DEC', 0))
    c_obj = SkyCoord(ra=obj_ra, dec=obj_dec, unit='deg')
    c_stars = SkyCoord(ra=df['ra'].values, dec=df['dec'].values, unit='deg')
    df['dist'] = c_stars.separation(c_obj).arcsec
    # 按角距离排序
    df = df.sort_values(by='dist')  # 按照 dist 列排序
    df.reset_index(drop=True, inplace=True)  # 重置索引
    df['obj_id'] = range(1, len(df) + 1)  # 重新设置 obj_id 列（从 1 开始）

    # 最终保存的csv文件只包含以下列：id, xcent, ycent, flux, peak, ra, dec, dist
    df = df[['obj_id', 'xcent', 'ycent', 'flux', 'peak', 'ra', 'dec', 'dist']]
    df['obj_id'] = df['obj_id'].astype(int)  # 确保 id 列是整数类型
    df['xcent'] = df['xcent'].astype(float)  # 将 xcent 列转换为整数类型
    df['ycent'] = df['ycent'].astype(float)  # 将 ycent 列转换为整数类型
    df['flux'] = df['flux'].astype(float)  # 确保 flux 列是浮点类型
    df['peak'] = df['peak'].astype(float)  # 确保 peak 列是浮点类型
    df['ra'] = df['ra'].astype(float)  # 确保 ra 列是浮点类型
    df['dec'] = df['dec'].astype(float)  # 确保 dec 列是浮点类型
    df['dist'] = df['dist'].astype(float)  # 确保 dist 列是浮点类型
    df.to_csv(pos_csvnm, index=False)  # 保存为 CSV 文件，包含列名
    # print(len(df))


def plot_stars(img, pos_csvnm, id_colnm='obj_id'):
    '''
    使用 DS9 圈出所找到的星
    '''
    import pyds9
    # print(pyds9.ds9_targets())
    d = pyds9.DS9()  # will open a new ds9 window or connect to an existing one
    d.set('width 2000')
    d.set('height 2000')
    d.set(f'file {img}')  # send the file to the open ds9 session
    d.set('zoom to fit')  # 适应窗口大小
    d.set('scale zscale')  # 设置 scale 为 zscale
    r_aper = 3
    df = pd.read_csv(pos_csvnm)
    for index, row in df.iterrows():  # 遍历筛选后的数据并打印 regions add circle 命令
        nid = row[id_colnm]
        nxi = row['xcent']
        nyi = row['ycent']
        # if row['obj'] != str(0):
        #     reg = f'image;circle({nxi},{nyi},{r_aper}) # width=4 color=red text={{{row['obj']}}}'  # example region
        #     d.set('regions', reg)  # load that region in ds9
        # else:
        reg = f'image;circle({nxi},{nyi},{r_aper}) # width=2 text={{{nid}}}'  # example region
        #     # reg += f'\nimage;point({nxi},{nyi}) # point=cross {30} width=2 color=red'  # example region
        d.set('regions', reg)  # load that region in ds9
    png_nm = os.path.splitext(os.path.abspath(pos_csvnm))[0] + '.png'
    # d.set(f'export png {png_nm}')  # 以高质量的方式导出 DS9 显示的内容 （只能导出一幅图）
    # print(pos_csvnm, png_nm)
    d.set(f'saveimage png {png_nm}')


def main():
    start_time = time.time()

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

    # **创建 pos 目录**
    pos_dir = 'pos'
    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)

    err_lst = []  # List to store images that cause errors
    for k in tqdm(range(len(imgs)), desc='Step 3. Finding Stars'):
        try:
            pos_csvnm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_pos.csv'  # 获取文件后缀
            pos_csv = os.path.join(pos_dir, pos_csvnm)
            findstars_sep(imgs[k], pos_csv)
            # plot_stars(imgs[k], pos_csv)
        except Exception as e:
            print(f"Error processing {imgs[k]}: {e}")
            err_lst.append(imgs[k])  # Add the current image to the error list

    # **记录错误文件以及成功处理的文件**
    if err_lst:
        with open('err2_finds.lst', 'a') as err_file:
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
