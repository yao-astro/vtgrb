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
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
from skimage.transform import AffineTransform

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

def match_stars(imgs):
    # 1. 读取所有csv
    poss = [i.replace('.fit', '_pos.csv') for i in imgs]  # 生成对应的 pos csv 文件名
    dfs = [pd.read_csv(f) for f in poss]

    # 2. 选某一帧为参考帧
    # ref_idx = len(dfs) // 2
    ref_idx = 0  # 选择第一帧作为参考帧
    ref_df = dfs[ref_idx].copy()
    ref_coords = np.vstack([ref_df['ra'], ref_df['dec']]).T
    ref_tree = cKDTree(ref_coords)
    ref_df['obj_idm'] = np.arange(1, len(ref_df)+1)

    # 3. 其他帧匹配
    match_radius = 1.0 / 3600  # 1 arcsec in deg
    for i, df in enumerate(dfs):
        obj_idm = np.zeros(len(df), dtype=int)
        if i == ref_idx:
            obj_idm = ref_df['obj_idm'].values
        else:
            coords = np.vstack([df['ra'], df['dec']]).T
            dists, idxs = ref_tree.query(coords, distance_upper_bound=match_radius)
            # 唯一性约束：同一参考星不能被多次匹配
            used = set()
            for j, (dist, idx) in enumerate(zip(dists, idxs)):
                if idx < len(ref_df) and dist < match_radius and idx not in used:
                    obj_idm[j] = ref_df['obj_idm'].iloc[idx]
                    used.add(idx)
                else:
                    obj_idm[j] = 0
        df['obj_idm'] = obj_idm
        outname = poss[i]
        # posm.csv只保留以下列
        keep_cols = ['obj_idm', 'obj_id', 'xcent', 'ycent', 'flux', 'peak', 'ra', 'dec', 'dist']
        df = df[keep_cols]
        # 不保存 obj_idm = 0 的行
        df = df[df['obj_idm'] != 0]
        df.to_csv(outname, index=False)
        # print(f'写入: {outname}')
    print('全部完成！')


def plot_grb_and_refs(fitnm, grb_x, grb_y, ref_stars, 
                      outdir='grb_match_plots', zoom_size=30, r_aper=3):
    markersize, elw = 10, 0.5
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # 读取图像数据
    with pyfits.open(fitnm) as hdul:
        img = hdul[0].data
    plt.figure(figsize=(6,6))
    # FITS像素坐标从1开始，Python数组从0开始，需-1
    print(grb_x, grb_y)
    x0, y0 = grb_x - 1, grb_y - 1
    x1, x2 = int(x0 - zoom_size), int(x0 + zoom_size)
    y1, y2 = int(y0 - zoom_size), int(y0 + zoom_size)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    subimg = img[y1:y2, x1:x2]
    norm = ImageNormalize(subimg, interval=ZScaleInterval(), stretch=SqrtStretch())
    # extent 方案，坐标轴直接为原图像素坐标
    plt.imshow(subimg, cmap='gray', origin='lower', norm=norm, extent=[x1, x2, y1, y2])
    # 标注GRB理论位置（红色空心圈，直接用原图像素坐标）
    r_pix = r_aper  # 你想要的像素半径，可自定义
    ax = plt.gca()
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_pix = bbox.width * fig.dpi
    pix2pt = (bbox.width * 72) / subimg.shape[1]  # 1像素对应多少points
    s_grb = (r_pix * pix2pt) ** 2
    plt.scatter([grb_x-1], [grb_y-1], s=s_grb, edgecolor='red', facecolor='none', 
                marker='o', linewidths=elw * 2, label='GRB', zorder=10)
    # 在红圈旁边标注其xy坐标
    plt.text(grb_x + 0.5, grb_y + 0.5, f"({grb_x:.4f}, {grb_y:.4f})", color='red', fontsize=9, zorder=11)
    # 标注参考星（原图像素坐标，无需换算）
    if len(ref_stars) > 0:
        r_pix_ref = 6  # 参考星像素半径，可自定义
        s_ref = (r_pix_ref * pix2pt) ** 2
        # 只保留在subimg范围内的参考星
        mask_in = (ref_stars['xcent']-1 >= x1) & (ref_stars['xcent']-1 < x2) & (ref_stars['ycent']-1 >= y1) & (ref_stars['ycent']-1 < y2)
        ref_stars_in = ref_stars[mask_in]
        plt.scatter(ref_stars_in['xcent']-1, ref_stars_in['ycent']-1, s=s_ref, edgecolor='green', facecolor='none', 
                    marker='o', label='Ref stars', zorder=9)
        for i, row in ref_stars_in.iterrows():
            plt.text(row['xcent']-1+3, row['ycent']-1+3, f"{int(row['obj_id'])}", 
                     color='green', fontsize=8)
    plt.title(os.path.basename(fitnm), fontsize=10)
    plt.xlabel('X [pix]')
    plt.ylabel('Y [pix]')
    plt.tight_layout()
    outpng = os.path.join(outdir, os.path.splitext(os.path.basename(fitnm))[0] + '_grb_match.png')
    plt.savefig(outpng, dpi=120)
    plt.close()


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

    # **检查有无 pos 目录**
    pos_dir = 'pos'
    if not os.path.exists(pos_dir):  # 如果不存在 pos 目录则不再继续
        print('No [pos] Dir, back to Step 3. Finding Stars')
        sys.exit(1)

    results = []
    for k in tqdm(range(len(imgs)), desc='Step 7. GRB Matching'):
        try:
            with pyfits.open(imgs[k]) as hdul:
                header = hdul[0].header
                obj_ra = float(header.get('OBJ_RA', 0))
                obj_dec = float(header.get('OBJ_DEC', 0))
                wcs = WCS(header)
                grb_x, grb_y = wcs.all_world2pix(obj_ra, obj_dec, 1)
        except Exception as e:
            print(f'Error reading FITS header or WCS for {imgs[k]}: {e}')
            continue

        # 自动选取目标附近的亮星
        pos_csvnm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_pos.csv'
        pos_csv = os.path.join(pos_dir, pos_csvnm)
        if not os.path.exists(pos_csv):
            print(f'Warning: {pos_csv} does not exist')
            continue
        df = pd.read_csv(pos_csv)
        if len(df) == 0:
            print(f'Warning: {pos_csv} 星表为空')
            continue
        # 计算所有星与GRB理论像素的距离，选最近的N颗亮星
        df['dist_pix'] = np.sqrt((df['xcent'] - grb_x)**2 + (df['ycent'] - grb_y)**2)
        # 只选取目标附近的亮星（如500像素内，最多20颗，且flux大于阈值）
        search_radius = 500  # 匹配半径
        flux_thresh = np.percentile(df['flux'], 70)  # 取前30%最亮的星作为亮星
        df_bright = df[(df['flux'] >= flux_thresh)]
        df_near = df_bright[df_bright['dist_pix'] < search_radius]
        ref_stars = df_near.nsmallest(20, 'dist_pix')
        if len(ref_stars) < 3:
            print(f'Warning: {pos_csv} 附近亮参考星太少，直接用WCS反投影')
            results.append({'frame': imgs[k], 'xcent': grb_x, 'ycent': grb_y, 'method': 'wcs'})
            continue
        # 用这些星的天球坐标和像素坐标做仿射变换
        src = np.vstack([ref_stars['ra'], ref_stars['dec']]).T
        dst = np.vstack([ref_stars['xcent'], ref_stars['ycent']]).T
        tform = AffineTransform()
        tform.estimate(src, dst)
        # 用仿射变换推算GRB理论天球坐标对应的像素坐标
        grb_xy = tform([[obj_ra, obj_dec]])[0]
        # 绘图标注
        plot_grb_and_refs(imgs[k], grb_xy[0], grb_xy[1], ref_stars)
        # 反算ra, dec（用当前帧的WCS）
        grb_ra, grb_dec = wcs.all_pix2world(grb_xy[0], grb_xy[1], 1)
        results.append({'frame': imgs[k], 'xcent': grb_xy[0], 'ycent': grb_xy[1], 'ra': grb_ra, 'dec': grb_dec, 'method': 'affine'})

    # 4. 保存结果
    df_out = pd.DataFrame(results)
    df_out.to_csv('grb_xyposs.csv', index=False)
    print('Saved GRB pixel coordinates for each frame to [grb_xyposs.csv]')

    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')


if __name__ == '__main__':
    main()