#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import yaml
import traceback
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')  # 放在导入 pyplot 之前
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm

sys.path.append(os.path.expanduser('~/astroWORK/codesyao/plot/'))
from pyplotsettings import set_mpl_style


def solve_quadratic(a, b, c, target_err):
    """
    解二次方程 a*x^2 + b*x + (c - target_err) = 0，求解对应的 x (即 mag) 值。
    
    参数：
    - a, b, c: 二次方程的系数
    - target_err: 目标的 mag_err 值
    
    返回：
    - mag 值的列表，包含解的数值
    """
    # 调整常数项，构造方程 a*x^2 + b*x + (c - target_err) = 0
    C = c - target_err
    
    # 计算判别式
    discriminant = b**2 - 4 * a * C
    
    if discriminant < 0:
        print("⚠️ 无实数解：拟合曲线不与 mag_err = 1/3 相交")
        return []  # 返回空列表，表示没有实数解
    else:
        sqrt_discriminant = np.sqrt(discriminant)
        # 计算两个解
        root1 = (-b + sqrt_discriminant) / (2 * a)
        root2 = (-b - sqrt_discriminant) / (2 * a)
        # 如果两个解相同，返回一个解，否则返回两个解
        if root1 == root2:
            return [root1]
        else:
            return [root1, root2]


def cal_maglimit(df, saved_csv, saved_pdf, 
                 mag_nm='mag', mag_err_nm='mag_err', plot_fig=True):
    """
    针对不同 r_aper，计算极限星等，返回包含 r_aper 和 mag_limit 的 DataFrame，并将所有 r_aper 的拟合图保存到同一个 PDF 文件。
    """
    from matplotlib.backends.backend_pdf import PdfPages
    maglim_list = []
    pdf = PdfPages(saved_pdf) if plot_fig else None
    for r_aper in sorted(df['r_aper'].unique()):
        try:
            df_sub = df[df['r_aper'] == r_aper].dropna(subset=[mag_nm, mag_err_nm])
            df_sub = df_sub[~df_sub[mag_nm].isin([np.inf, -np.inf])]
            df_sub = df_sub[~df_sub[mag_err_nm].isin([np.inf, -np.inf])]
            df_sub = df_sub[df_sub[mag_err_nm] < 1]  # 过滤掉 mag_err >= 1 的数据
            if len(df_sub) < 5:
                continue  # 数据太少跳过
            df_fit = df_sub[df_sub[mag_err_nm] >= 0.15]  # 只取 mag_err >= 0.15 的数据进行拟合
            if df_fit.empty:
                continue  # 如果没有足够的数据进行拟合，则跳过

            # **拟合**
            mag, mag_err = df_fit[mag_nm], df_fit[mag_err_nm]  # 取出需要拟合的数据
            coeffs = np.polyfit(mag, mag_err, 2)
            a, b, c = coeffs
            mag_fit = np.linspace(min(df_sub[mag_nm]), max(df_sub[mag_nm]), 200)
            mag_err_fit = a * mag_fit**2 + b * mag_fit + c
            mag_lim_roots = solve_quadratic(a, b, c, 1 / 3)
            if len(mag_lim_roots) == 0:
                mag_lim = np.nan
            else:
                mag_lim = np.max(mag_lim_roots)

            maglim_list.append({'r_aper': r_aper, 'mag_limit': mag_lim})

            # **绘图**
            if plot_fig:
                fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                ax.errorbar(df_sub[mag_nm], df_sub[mag_err_nm], fmt='+', c='grey')
                # ax.errorbar(mag, mag, fmt='+', c='k')
                mask = mag_err_fit < 1
                ax.plot(mag_fit[mask], mag_err_fit[mask], color='red', 
                        label='Fitted Curve', zorder=10)
                ax.axhline(1 / 3, ls='--', color='green')
                if not np.isnan(mag_lim):
                    ax.axvline(mag_lim, ls='--', color='blue', 
                               label=f'3 sigma mag limit: {mag_lim:.2f}')
                ax.legend(loc='upper left', fontsize=24)
                ax.set_xlabel('Mag')
                ax.set_ylabel(r'Mag$_{\rm err}$')
                # ax.set_title(f'r_aper={r_aper}')
                ax.text(0.03, 0.87, fr'$r_{{\rm aper}}={r_aper}$', 
                        transform=ax.transAxes, fontsize=24, va='top', ha='left')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        except Exception as e:
            print(f"r_aper={r_aper} Fit Error: {e}")
            mag_lim = np.nan
    if plot_fig:
        pdf.close()
    maglim_df = pd.DataFrame(maglim_list)
    maglim_df.to_csv(saved_csv, index=False)
        

def main():
    start_time = time.time()
    set_mpl_style()
    markersize, elw = 10, 0.5  # 误差棒的线宽

    # **读取参数配置文件**
    config_filenm = sys.argv[1]
    with open(config_filenm, 'r') as f:
        config = yaml.safe_load(f)
    # target_ra, target_dec = config['target_radec']  # 读取目标天球坐标 (RA, Dec)
    # target_nm = config['target_nm']  # 目标名称
    raper_chos = config['raper_chos']
    r_cho, r_in, r_out, r_step, r_all = raper_chos
    # print(r_cho, r_in, r_out, r_step, r_all)
    # t0 = pd.to_datetime(config['t0'])
    # print(t0)

    # 生成所有r对，仅使用 r_cho 和 r_all 的对
    r_pairs = [[r_cho, r_all]]

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
    # print(imgs)
    if not imgs:  # 检查 imgs 是否为空（如果不为空则继续执行，如果为空则不再执行）
        print(f'Error [{os.path.basename(__file__)}]: {fit_lstnm} is empty, Exit...')
        sys.exit(1)

    # **检查有无 aphot 目录**
    aphot_dir = 'aphot'
    if not os.path.exists(aphot_dir):  # 如果不存在 aphot 目录则不再继续）
        print('No [aphot] Dir, back to Step 4. Aperture Photometry')
        sys.exit(1)

    # **检查有无 maglim 目录**
    maglim_dir = 'maglim'
    if not os.path.exists(maglim_dir):  # 如果不存在 maglim 目录则创建
        os.makedirs(maglim_dir)

    # 画 mag - mag_err 图
    for k in tqdm(range(len(imgs)), desc='Step 7. Calculate Magnitude Limit'):
        try:
            aphot_parqnm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_aphot.parquet'
            maglim_csvnm = aphot_parqnm.replace('_aphot.parquet', '_maglim.csv')
            maglim_pdfnm = aphot_parqnm.replace('_aphot.parquet', '_maglim.pdf')

            aphot_parq = os.path.join(aphot_dir, aphot_parqnm)
            maglim_csv = os.path.join(maglim_dir, maglim_csvnm)  # 保存到 maglim_dir
            maglim_pdf = os.path.join(maglim_dir, maglim_pdfnm)  # 保存到 maglim_dir

            # 计算 mag_limit 并且保存结果
            df = pd.read_parquet(aphot_parq)
            cal_maglimit(df, maglim_csv, maglim_pdf)

        except Exception as e:
            print(f"Error processing {aphot_parqnm}: {e}")

    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')


if __name__ == "__main__":
    main()
