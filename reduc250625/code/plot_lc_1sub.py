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



def plot_lc(lc_csvnm, target_nm, t0=None):
    '''
    画光变曲线
    时间线性图中两个波段的数据分开画，并且最底下画出 B-R；
    时间 log 图中两个波段的数据画到一起，最底下画出 B-R
    （后续补充 B-R）
    区分单帧以及图像合并后的数据点
    '''
    set_mpl_style()
    markersize, elw = 8, 0.5  # 误差棒的线宽

    df = pd.read_csv(lc_csvnm)
    dfo = df[df['obj_id'] == target_nm]
    r_lst = df['r_aper'].unique()
    band_lst = dfo['band'].unique()

    # **画时间为线性的图**
    fig, ax = plt.subplots(len(band_lst), 1, sharex=True, figsize=(16, 20))
    fig.subplots_adjust(wspace=0, hspace=0.02)
    fig.align_ylabels(ax)  # 对齐y轴标签
    if len(band_lst) == 1:  # 如果只有一个子图，则ax是单个对象，需要转换为列表
        ax = [ax]
    for i in range(len(band_lst)):
        dfb = dfo[dfo['band'] == band_lst[i]]
        for is_single in [True, False]:  # True -> ncombine == 1, False -> ncombine != 1
            dfb_sub = dfb[dfb['ncombine'] == 1] if is_single else dfb[dfb['ncombine'] != 1]
            if dfb_sub.empty:
                continue
            t_obs = pd.to_datetime(dfb_sub['t_obs'])
            t_obs_err = pd.to_timedelta(dfb_sub['t_obs_err'], unit='s')  # 将 t_obs_err 从秒转换为 timedelta 对象，作为 xerr
            fy, fy_err = dfb_sub['mag_a'], dfb_sub['mag_a_err']
            bkg_median = dfb_sub['adu_bkg_median']
            # **清洗 NaN 数据**
            mask_nan = ~np.isnan(t_obs) & ~np.isnan(fy_err) & ~np.isnan(fy)
            t_obs, t_obs_err = t_obs[mask_nan], t_obs_err[mask_nan]
            fy, fy_err = fy[mask_nan], fy_err[mask_nan]
            bkg_median = bkg_median[mask_nan]
            # **区分两种数据点风格**
            # custom_colors = ['blue', 'red']
            custom_colors = ['C0', 'C1']
            color_val = f'C{i}' if is_single else custom_colors[i]
            marker_style = 'o' if is_single else 'D'
            alpha_val = 0.5 if is_single else 0.9
            markersize_val = markersize if is_single else markersize * 0.75
            elw_val = elw if is_single else elw * 2
            capsize_val = 0 if is_single else elw * 5
            label_tag = 'single' if is_single else 'stacked'
            zorder_val = 1 if is_single else 3
            # ax[i].errorbar(t_obs, fy, xerr=t_obs_err, yerr=fy_err, 
            #                fmt=marker_style, color=color_val, alpha=alpha_val,
            #                markersize=markersize_val, elinewidth=elw_val, 
            #                capsize=capsize_val, label=label_tag, zorder=zorder_val)
            for j in range(len(fy)):
                x = t_obs.iloc[j]
                y = fy.iloc[j]
                xerr = t_obs_err.iloc[j]
                yerr = fy_err.iloc[j]
                label_tag_val = label_tag if j == 0 else None  # 只在第一个点加图例
                if yerr == 0:
                    ax[i].errorbar(x, y, xerr=xerr,  # 如果是上限，先画数据点
                                   fmt=marker_style, color=color_val, alpha=alpha_val, 
                                   markersize=markersize_val, elinewidth=elw_val, 
                                   zorder=zorder_val)
                    arrow_len = 0.5  # 箭头长度，y轴单位长度
                    ax[i].arrow(x, y, 0, arrow_len,  # 画箭头
                                length_includes_head=True,
                                head_width=arrow_len / 5,  # ← 箭头横向宽度（更粗）
                                head_length=arrow_len / 3,  # ← 箭头纵向高度
                                fc=color_val, ec=color_val, alpha=alpha_val,
                                zorder=zorder_val)
                else:  # 正常误差条
                    ax[i].errorbar(x, y, xerr=xerr, yerr=yerr,
                                   fmt=marker_style, color=color_val, alpha=alpha_val,
                                   markersize=markersize_val, elinewidth=elw_val, 
                                   capsize=capsize_val, zorder=zorder_val)

        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%dT%H:%M'))
        plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=16)
        # fy_lim = cal_ylim(fy, fy_err)
        # ax[i].set_ylim(fy_lim[0], fy_lim[1])
        ax[i].set_ylabel('Mag')
        ax[i].invert_yaxis()
        ax[i].grid(True, linestyle='--', alpha=0.6)  # 可选：设置虚线网格，透明度 0.6

        # **标注波段**
        band_text = f'{target_nm} ({band_lst[i]})'
        ax[i].text(0.97, 0.9, band_text, transform=ax[i].transAxes, 
                   color=f'C{i}', fontsize=30., fontweight='bold', 
                   fontfamily='serif', horizontalalignment='right')

    ax[-1].set_xlabel('Obs. Time')
    # ** 标注 T0 和 r_aper**
    if t0:
        t0_str = t0.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # 只保留毫秒 3 位
        t0r_text = r'$T_0$={}, r={}'.format(t0_str, r_lst[0])
    else:
        t0r_text = f'r = {r_lst[0]}'
    ax[-1].text(0.03, 0.06, t0r_text, transform=ax[-1].transAxes, 
                color='k', fontsize=24., fontweight='bold', 
                fontfamily='serif', horizontalalignment='left')

    lc_pdfnm = lc_csvnm.replace('.csv', '.pdf')
    plt.savefig(lc_pdfnm, format='pdf', dpi=1200, 
                orientation='landscape', bbox_inches='tight')
    # plt.show()
    plt.close()

    # **画时间减 t0 后 log 的图**
    if t0:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(16, 10))
        for i in range(len(band_lst)):
            dfb = dfo[dfo['band'] == band_lst[i]]
            for is_single in [True, False]:
                dfb_sub = dfb[dfb['ncombine'] == 1] if is_single else dfb[dfb['ncombine'] != 1]
                if dfb_sub.empty:
                    continue
                t_obs = pd.to_datetime(dfb_sub['t_obs'])
                t_obs_err = pd.to_timedelta(dfb_sub['t_obs_err'], unit='s')
                fy, fy_err = dfb_sub['mag_a'], dfb_sub['mag_a_err']
                bkg_median = dfb_sub['adu_bkg_median']
                # **清洗 NaN 数据**
                mask_nan = ~np.isnan(t_obs) & ~np.isnan(fy_err) & ~np.isnan(fy)
                t_obs, t_obs_err = t_obs[mask_nan], t_obs_err[mask_nan]
                fy, fy_err = fy[mask_nan], fy_err[mask_nan]
                bkg_median = bkg_median[mask_nan]
                t_diff = (t_obs - t0).dt.total_seconds()  # 时间转为秒数
                t_obs_err = t_obs_err.dt.total_seconds()  # 时间误差需要和时间保持一个格式

                # **区分两种数据点风格**
                # custom_colors = ['blue', 'red']
                # custom_colors = ['cyan', 'red']
                custom_colors = ['C0', 'C1']
                color_val = f'C{i}' if is_single else custom_colors[i]
                marker_style = 'o' if is_single else 'D'
                alpha_val = 0.4 if is_single else 0.8
                markersize_val = markersize if is_single else markersize * 0.75
                elw_val = elw if is_single else elw * 2
                capsize_val = 0 if is_single else elw * 5
                label_tag = f'{band_lst[i]}-single' if is_single else f'{band_lst[i]}-stacked'
                zorder_val = 1 if is_single else 3
                # **如果 yerr = 0，则只画上限值，同时画出数据点向下的箭头，其他点则正常画误差棒**
                for j in range(len(fy)):
                    x = t_diff.iloc[j]
                    y = fy.iloc[j]
                    xerr = t_obs_err.iloc[j]
                    yerr = fy_err.iloc[j]
                    label_tag_val = label_tag if j == 0 else None  # 只在第一个点加图例
                    if yerr == 0:
                        ax.errorbar(x, y, xerr=xerr,  # 如果是上限，先画数据点
                                    fmt=marker_style, color=color_val, alpha=alpha_val, 
                                    markersize=markersize_val, elinewidth=elw_val,
                                    zorder=zorder_val)
                        arrow_len = 0.3  # 箭头长度，y轴单位长度
                        ax.arrow(x, y, 0, arrow_len,  # 画箭头
                                 length_includes_head=True,
                                 head_width=16 * xerr,  # ← 箭头横向宽度（更粗）
                                 head_length=arrow_len / 3,  # ← 箭头纵向高度
                                 fc=color_val, ec=color_val, alpha=alpha_val,
                                 zorder=zorder_val)
                    else:  # 正常误差条
                        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                                    fmt=marker_style, color=color_val, alpha=alpha_val,
                                    markersize=markersize_val, elinewidth=elw_val, 
                                    capsize=capsize_val, label=label_tag_val, zorder=zorder_val)
                ax.legend(loc=[0.03, 0.1], fontsize=16., ncol=len(band_lst), framealpha=0.5)
                fx_lim = cal_xloglim(t_diff)
                # print(fx_lim)
                # ax.set_xlim(fx_lim[0], fx_lim[1])
                # fy_lim = cal_ylim(fy, fy_err)
                # ax.set_ylim(fy_lim[0], fy_lim[1])

        ax.set_ylabel('Mag')
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)  # 可选：设置虚线网格，透明度 0.6
        # **标注 target_nm**
        ax.text(0.97, 0.94, f'{target_nm}', transform=ax.transAxes, 
                color='k', fontsize=30., fontweight='bold', 
                fontfamily='serif', horizontalalignment='right')
        ax.set_xlabel(r'$T - T_0$ (Sec.)')
        # **标注 T0 和 r_aper**
        t0_str = t0.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # 只保留毫秒 3 位
        t0r_text = r'$T_0$={}, r={}'.format(t0_str, r_lst[0])
        ax.text(0.03, 0.04, t0r_text, transform=ax.transAxes, 
                color='k', fontsize=24., fontweight='bold', 
                fontfamily='serif', horizontalalignment='left')

        lc_pdfnm = lc_csvnm.replace('.csv', '_log.pdf')
        plt.savefig(lc_pdfnm, format='pdf', dpi=1200, 
                    orientation='landscape', bbox_inches='tight')
        lc_pngnm = lc_csvnm.replace('.csv', '_log.png')
        plt.savefig(lc_pngnm, format='png', dpi=600, 
                    orientation='landscape', bbox_inches='tight')
        # plt.show()
        plt.close()


def plot_lc_html(lc_csvnm, target_nm, t0=None):
    '''
    画光变曲线并输出为 HTML 格式
    两个波段上下两幅图，横轴时间对齐，中间有显著分隔
    上图无x轴标签，保留图例，数据点hover显示为“mag±err一行，时间一行”
    左上角标注 mag 均值±标准差
    并在每个子图中画出该波段的统计点（中值/均值）
    '''
    import plotly.graph_objects as go
    import plotly.subplots as sp
    import pandas as pd
    import numpy as np

    markersize, elw = 8, 0.5  # 误差棒的线宽
    df = pd.read_csv(lc_csvnm)
    r_lst = df['r_aper'].unique()
    band_lst = df['band'].unique()

    # 统计信息（集成stat_lc.py功能）
    stat_rows = []
    for band, dfg in df.groupby('band'):
        mag_median = dfg['mag'].median()
        mag_std = dfg['mag'].std()
        t_start = pd.to_datetime(dfg['t_start'])
        tmin = t_start.min()
        tmax = t_start.max()
        exptime = dfg['EXPTIME'].iloc[0] if 'EXPTIME' in dfg.columns else 0
        tmax_plus_exp = tmax + pd.to_timedelta(exptime, unit='s')
        t_center = tmin + (tmax_plus_exp - tmin) / 2
        t_span = (tmax_plus_exp - tmin) / 2
        stat_rows.append({
            'band': band,
            't_center': t_center.strftime('%Y-%m-%dT%H:%M:%S'),
            't_span_sec': t_span.total_seconds(),
            'mag_median': mag_median,
            'mag_std': mag_std
        })
    stat_dict = {row['band']: row for row in stat_rows}
    # 保存统计点为新csv
    stat_csv = lc_csvnm.replace('.csv', '_stat.csv')
    pd.DataFrame(stat_rows).to_csv(stat_csv, index=False)
    print(f"统计点已保存为 {stat_csv}")

    # 创建上下两行子图，x轴对齐
    fig = sp.make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.18,  # 增大分隔
        subplot_titles=("", "")  # 不显示 band 名称
    )

    color_map = ['blue', 'red', 'orange', 'green']
    stat_color_map = {'VT_B': 'cyan', 'VT_R': 'orange'}
    for i, band in enumerate(band_lst):
        dfb = df[df['band'] == band]
        # 计算均值和标准差
        mag_mean = dfb['mag'].mean()
        mag_std = dfb['mag'].std()
        # 左上角标注
        xref = 'x domain' if i == 0 else 'x2 domain'
        yref = 'y domain' if i == 0 else 'y2 domain'
        fig.add_annotation(
            text=f"{mag_mean:.3f} ± {mag_std:.3f}",
            xref=xref, yref=yref,
            x=0.01, y=0.99, showarrow=False,
            font=dict(size=18, color=color_map[i % len(color_map)]),
            align="left",
            row=i+1, col=1
        )
        for is_single in [True, False]:
            dfb_sub = dfb[dfb['ncombine'] == 1] if is_single else dfb[dfb['ncombine'] != 1]
            if dfb_sub.empty:
                continue
            t_start = pd.to_datetime(dfb_sub['t_start'])
            fy = dfb_sub['mag'].round(3)
            fy_err = dfb_sub['mag_err']
            color_val = color_map[i % len(color_map)] if is_single else 'cyan'
            size_val = markersize * 2 if is_single else markersize * 2.5
            symbol_val = 'circle' if is_single else 'diamond'
            # hover显示mag±err和时间
            hovertemplate = "%{y:.3f} ± %{customdata:.3f}<br>%{x|%Y-%m-%dT%H:%M:%S}<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=t_start, y=fy,
                    error_y=dict(
                        type='data',
                        array=fy_err,
                        visible=True,
                        thickness=elw * 2
                    ),
                    customdata=fy_err,
                    mode='markers',
                    marker=dict(
                        color=color_val,
                        size=size_val,
                        symbol=symbol_val,
                        line=dict(width=1, color='black')
                    ),
                    name=f'{band} {"single" if is_single else "stacked"}',
                    showlegend=True,
                    hovertemplate=hovertemplate
                ),
                row=i+1, col=1
            )
        # # 画统计点（中值），颜色区分波段
        # stat = stat_dict[band]
        # stat_color = stat_color_map.get(band, 'black')
        # fig.add_trace(
        #     go.Scatter(
        #         x=[stat['t_center']],
        #         y=[stat['mag_median']],
        #         error_y=dict(type='data', array=[stat['mag_std']], visible=True, thickness=elw * 2),
        #         error_x=dict(type='data', array=[stat['t_span_sec']], visible=True, thickness=elw * 2),
        #         mode='markers',
        #         marker=dict(color=stat_color, size=18, symbol='star'),
        #         name=f'{band} stat',
        #         showlegend=True,
        #         hovertemplate=f"统计点<br>mag={stat['mag_median']:.3f}±{stat['mag_std']:.3f}<br>time=%{{x|%Y-%m-%dT%H:%M:%S}}±{stat['t_span_sec']:.0f}s<extra></extra>"
        #     ),
        #     row=i+1, col=1
        # )
        # # 画中值和标准差的三条虚线
        # mag_median = dfb['mag'].median()
        # mag_std = dfb['mag'].std()
        # for offset, dash, color in zip([0, +mag_std, -mag_std], ['dash', 'dot', 'dot'], ['gray', 'gray', 'gray']):
        #     fig.add_hline(
        #         y=mag_median + offset,
        #         line=dict(color=color, width=2, dash=dash),
        #         row=i+1, col=1
        #     )

    # 设置标题和布局
    if t0:
        t0_str = t0.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        t0r_text = f'T₀ = {t0_str}, r = {r_lst[0]}'
    else:
        t0r_text = f'r = {r_lst[0]}'
    fig.update_layout(
        title=dict(
            text=f"<b>Light Curve of {target_nm}</b><br><sub>{t0r_text}</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",  # 上图无x轴标签
            tickformat="%y-%m-%dT%H:%M:%S",
            showgrid=True,
            tickangle=45,
            tickfont=dict(size=14)
        ),
        xaxis2=dict(
            title="Obs. Time",  # 仅下图有x轴标签
            tickformat="%y-%m-%dT%H:%M:%S",
            showgrid=True,
            tickangle=45,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title="Mag",
            showgrid=True,
            autorange="reversed"
        ),
        yaxis2=dict(
            title="Mag",
            showgrid=True,
            autorange="reversed"
        ),
        template="plotly_dark",
        width=1600,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 在两图之间添加显著分隔线
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, x1=1, y0=0.495, y1=0.505,
        fillcolor="gray", opacity=0.5, layer="below", line_width=0,
    )

    lc_htmlnm = lc_csvnm.replace('.csv', '.html')
    fig.write_html(lc_htmlnm)
    print(f"Light curve plot saved to {lc_htmlnm}")


def filter_outliers(df, mag_col='mag', mag_err_col='mag_err', sigma=3, max_mag_err=0.2, max_iter=5):
    '''
    剔除mag离群点（sigma剪裁）和mag_err过大的点，支持多次迭代
    '''
    if mag_col not in df.columns or mag_err_col not in df.columns:
        return df
    df_filt = df.copy()
    for _ in range(max_iter):
        mag = df_filt[mag_col]
        mag_err = df_filt[mag_err_col]
        med = mag.median()
        std = mag.std()
        mask = (np.abs(mag - med) < sigma * std) & (mag_err < max_mag_err)
        if mask.sum() == len(df_filt):
            break
        df_filt = df_filt[mask]
    return df_filt


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

    lc_path = 'lc/'  # write data to lc
    os.makedirs(lc_path, exist_ok=True)

    df = pd.read_csv('std_aphot.csv')
    df_sel = df[df['r_aper'] == r_cho]  # 指定孔径半径
    # 剔除离群点和大误差点
    df_sel = filter_outliers(df_sel, mag_col='mag', mag_err_col='mag_err', sigma=3, max_mag_err=0.2, max_iter=5)
    # 保存为新csv
    r_cho_text = f'{r_cho:.1f}'.replace('.', '_')
    lc_csvnm = os.path.join(lc_path, f'{target_nm}_lc_{r_cho_text}.csv')
    df_sel.to_csv(lc_csvnm, index=False)
    print(f'已保存为 {lc_csvnm}，包含 {len(df_sel)} 行。')
    # plot_lc(lc_csvnm, target_nm, t0)
    plot_lc_html(lc_csvnm, target_nm)

    # Cal Time
    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')


if __name__ == "__main__":
    main()
