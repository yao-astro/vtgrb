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


def iterative_fit(x, y, xerr, yerr, max_iter=5, threshold=3.0):
    '''
    迭代剔除偏离点并拟合
    '''
    def linear_func(beta, x):
        '''
        定义线性函数 y = x + c
        '''
        return x + beta[0]

    for i in range(max_iter):
        # 使用 ODR 进行拟合
        model = odr.Model(linear_func)
        data = odr.RealData(x, y, sx=xerr, sy=yerr)
        odr_fitter = odr.ODR(data, model, beta0=[0])
        output = odr_fitter.run()
        c = output.beta[0]  # 截距
        c_err = output.sd_beta[0]  # 截距误差
        # 计算残差
        y_fit = x + c
        residuals = np.abs(y - y_fit)  # 计算每个点的残差
        sigma = np.std(residuals)  # 计算标准差
        # 剔除偏离超过 `threshold * sigma` 的点
        mask = residuals <= threshold * sigma
        if np.all(mask):  # 如果没有剔除点，则提前停止迭代
            break
        x, y, xerr, yerr = x[mask], y[mask], xerr[mask], yerr[mask]
    return x, y, xerr, yerr, c, c_err


def get_magc(aphot_filenm, r_pair, mag_threshold=20, mag_err_threshold=0.5, plot_fig=True):
    '''
    获取孔径改正值，对比每颗星在小孔径 r_pair[0] 和大孔径 r_pair[1] 时的星等，直线拟合出截距作为星等改正值
    r_pair: [r_cho, r_ref] -- 要求是一个包含两个孔径的列表
    '''
    df = pd.read_parquet(aphot_filenm)  # 读取全图孔径测光的 CSV/Parquet 文件
    # 选择 r_aper=r_pair[0] 和 r_aper=r_pair[1] 的数据
    df_rcho = df[df['r_aper'] == r_pair[0]].set_index('id')
    df_rref = df[df['r_aper'] == r_pair[1]].set_index('id')
    common_stars = df_rcho.index.intersection(df_rref.index)  # 确保索引匹配
    df_rcho = df_rcho.loc[common_stars]
    df_rref = df_rref.loc[common_stars]
    # 数据筛选
    valid_mask = (
        (df_rcho['mag_err'] > 0) & (df_rref['mag_err'] > 0) & 
        (df_rcho['mag_err'] < mag_err_threshold) & (df_rref['mag_err'] < mag_err_threshold) &
        (df_rcho['mag'] < mag_threshold) & (df_rref['mag'] < mag_threshold)
    )
    df_rcho = df_rcho[valid_mask]
    df_rref = df_rref[valid_mask]

    # **获取数据并拟合**
    x, x_err = df_rcho['mag'], df_rcho['mag_err']
    y, y_err = df_rref['mag'], df_rref['mag_err']
    x_fit, y_fit, xfit_err, yfit_err, c, c_err = iterative_fit(x, y, x_err, y_err)  # 进行迭代拟合
    # print(c, c_err)

    # **绘图**
    if plot_fig:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 10))
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, 
                    fmt='o', capsize=3, alpha=0.4, color='grey')  # 绘制原始数据散点图
        ax.errorbar(x_fit, y_fit, xerr=xfit_err, yerr=yfit_err, 
                    fmt='o', capsize=3, alpha=0.7, color='C0')  # 绘制拟合散点图
        # 绘制拟合直线
        x_line = np.linspace(min(x_fit) - 0.5, max(x_fit) + 0.5, 100)
        y_line = x_line + c
        ax.plot(x_line, y_line, 'C1-', label=f'Fit: y = x + {c:.2f}')
        ax.set_xlabel(f'Magnitude at r_aper = {r_pair[0]}')
        ax.set_ylabel(f'Magnitude at r_aper = {r_pair[1]}')

        magc_text = r'$m_{{{:.1f}}}=m_{{{:.1f}}} {:.3f} (\pm{:.3f})$'.format(r_pair[1], r_pair[0], c, c_err)
        ax.text(0.96, 0.06, magc_text, transform=ax.transAxes, 
                color='C1', fontsize=24., fontweight='bold', 
                fontfamily='serif', horizontalalignment='right')
        ax.text(0.03, 0.96, aphot_filenm, transform=ax.transAxes, 
                color='k', fontsize=12., fontweight='bold', 
                fontfamily='serif', horizontalalignment='left')

        plt.grid(True, linestyle='--', alpha=0.5)
        # magc_pdfnm = os.path.splitext(aphot_csvnm)[0] + '_magc.pdf'
        # plt.savefig(magc_pdfnm, format='pdf', dpi=300, 
        #             orientation='landscape', bbox_inches='tight')
        # plt.show()
        # plt.close()
    else:
        fig = None  # 如果不需要绘图，则返回 None

    return c, c_err, fig


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

    # ****计算孔径测光半径列表****
    rlst = np.round(np.arange(r_step, r_in + r_step, r_step), 1).tolist()
    # 生成所有r对，第一个小于第二个
    r_pairs = [[r1, r2] for i, r1 in enumerate(rlst) for r2 in rlst[i+1:]]
    r_pairs = [[r_cho, r_all]]  # 仅使用 r_cho 和 r_all 的对
    # print(type(r_pairs))

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
    
    # **检查有无 aphot 目录**
    aphot_dir = 'aphot'
    if not os.path.exists(aphot_dir):  # 如果不存在 aphot 目录则不再继续）
        print('No [aphot] Dir, back to Step 4. Aperture Photometry')
        sys.exit(1)
    
    # **检查有无 magc 目录**
    magc_dir = 'magc'
    if not os.path.exists(magc_dir):  # 如果不存在 magc 目录则创建
        os.makedirs(magc_dir)

    err_lst = []  # List to store images that cause errors
    for k in tqdm(range(len(imgs)), desc='Step 6. Correcting Magnitude'):
        try:
            aphot_parqnm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_aphot.parquet'
            magc_csvnm = aphot_parqnm.replace('_aphot.parquet', '_magc.csv')
            magc_pngnm = aphot_parqnm.replace('_aphot.parquet', '_magc.png')

            aphot_parq = os.path.join(aphot_dir, aphot_parqnm)
            magc_csv = os.path.join(magc_dir, magc_csvnm)  # 保存到 magc_dir
            magc_png = os.path.join(magc_dir, magc_pngnm)  # 保存到 magc_dir
            # 保存所有r对的校正量和误差到csv，所有图像到png
            magc_rows = []
            for j in range(len(r_pairs)):
                c, c_err, fig = get_magc(aphot_parq, r_pairs[j])
                magc_rows.append({'r1': r_pairs[j][0], 'r2': r_pairs[j][1], 'c': c, 'c_err': c_err})
                fig.savefig(magc_png, dpi=150, bbox_inches='tight')
                plt.close(fig)
            magc_df = pd.DataFrame(magc_rows)
            magc_df.to_csv(magc_csv, index=False)
        except:
            traceback.print_exc()
            err_lst.append(imgs[k])  # Add the current image to the error list

    # **记录错误文件以及成功处理的文件**
    if err_lst:
        with open('err6_corma.lst', 'a') as err_file:
            for img in err_lst:
                err_file.write(f'{img}\n')
    suc_lst = [img for img in imgs if img not in err_lst]
    with open('suc.lst', 'w') as suc_file:
        for img in suc_lst:
            suc_file.write(f'{img}\n')

    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.3f} s')


if __name__ == "__main__":
    main()
