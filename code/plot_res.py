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


def cal_xloglim(fx, nlim0=1.5, nlim1=1.5):
    '''
    计算 x 轴范围（log 轴）
    nlim0，nlim1 分别为左右留白系数
    '''
    fx_lim = (np.min(fx) / nlim0, np.max(fx) * nlim1)  # 增加留白
    return fx_lim


def cal_ylim(fy, fy_err, arrowlen=0.33, nlim0=0.5, nlim1=0.5):
    '''
    计算合理的 ylim，自动跳过 NaN
    '''
    # 若fy_err有'mag_limit'字符串，则替换为arrowlen
    fy = np.asarray(fy, dtype='float64')
    fy_err_arr = np.array(fy_err)
    if fy_err_arr.dtype.kind in {'U', 'S', 'O'}:
        # 字符串型，替换'mag_limit'为arrowlen，其余转float
        fy_err_arr = np.array([arrowlen if (str(x).strip().lower() == 'mag_limit') else x for x in fy_err_arr])
    fy_err = np.asarray(fy_err_arr, dtype='float64')
    mask = ~np.isnan(fy) & ~np.isnan(fy_err)
    if not np.any(mask):
        return (0, 1)  # 全为NaN时返回默认范围
    fy_valid = fy[mask]
    fy_err_valid = fy_err[mask]
    fy_min = np.min(fy_valid - fy_err_valid)
    fy_max = np.max(fy_valid + fy_err_valid)
    fy_margin = fy_max - fy_min
    fy_lim = (fy_min - fy_margin * nlim0, fy_max + fy_margin * nlim1)
    return fy_lim


def plot_lc(lc_csvnm, target_nm, t0=None):
    '''
    画光变曲线
    时间线性图中两个波段的数据分开画，并且最底下画出 B-R；
    时间 log 图中两个波段的数据画到一起，最底下画出 B-R
    （后续补充 B-R）
    区分单帧以及图像合并后的数据点
    '''
    set_mpl_style()
    markersize, elw = 14, 0.75  # 误差棒的线宽

    df = pd.read_csv(lc_csvnm)
    r_lst = df['r_aper'].unique()
    band_lst = df['band'].unique()

    # # **画时间为线性的图**
    # fig, ax = plt.subplots(len(band_lst), 1, sharex=True, figsize=(16, 20))
    # fig.subplots_adjust(wspace=0, hspace=0.02)
    # fig.align_ylabels(ax)  # 对齐y轴标签
    # if len(band_lst) == 1:  # 如果只有一个子图，则ax是单个对象，需要转换为列表
    #     ax = [ax]
    # for i in range(len(band_lst)):
    #     dfb = df[df['band'] == band_lst[i]]
    #     for is_single in [True, False]:  # True -> ncombine == 1, False -> ncombine != 1
    #         dfb_sub = dfb[dfb['ncombine'] == 1] if is_single else dfb[dfb['ncombine'] != 1]
    #         if dfb_sub.empty:
    #             continue
    #         # 使用 t_start 和 t_end 计算横坐标和误差
    #         t_start = pd.to_datetime(dfb_sub['t_start'])
    #         t_end = pd.to_datetime(dfb_sub['t_end'])
    #         t_obs = t_start + (t_end - t_start) / 2  # 观测时间中心
    #         t_obs_err = (t_end - t_start) / 2        # 观测时间误差
    #         fy, fy_err = dfb_sub['mag'], dfb_sub['mag_err']
    #         fy_err = pd.to_numeric(fy_err, errors='coerce')  # 强制转为 float，非数值变 nan
    #         # **区分两种数据点风格**
    #         # custom_colors = ['blue', 'red']
    #         custom_colors = ['C0', 'C1']
    #         color_val = f'C{i}' if is_single else custom_colors[i]
    #         marker_style = 'o' if is_single else 'D'
    #         alpha_val = 0.5 if is_single else 0.9
    #         markersize_val = markersize if is_single else markersize * 0.75
    #         elw_val = elw if is_single else elw * 2
    #         capsize_val = 0 if is_single else elw * 5
    #         label_tag = 'single' if is_single else 'stacked'
    #         zorder_val = 1 if is_single else 3
    #         # ax[i].errorbar(t_obs, fy, xerr=t_obs_err, yerr=fy_err, 
    #         #                fmt=marker_style, color=color_val, alpha=alpha_val,
    #         #                markersize=markersize_val, elinewidth=elw_val, 
    #         #                capsize=capsize_val, label=label_tag, zorder=zorder_val)
    #         # 转为 matplotlib 支持的 float
    #         x = mdates.date2num(t_obs)
    #         xerr = np.array([td.total_seconds()/(24*3600) for td in t_obs_err])  # xerr 单位需为天
    #         for j in range(len(fy)):
    #             xj = x[j]
    #             xerrj = xerr[j]
    #             y = fy.iloc[j]
    #             yerr = fy_err.iloc[j]
    #             t0j = t_start.iloc[j]
    #             t1j = t_end.iloc[j]
    #             label_tag_val = label_tag if j == 0 else None  # 只在第一个点加图例
    #             if np.isnan(yerr):  # mag_limit点，画marker+箭头
    #                 # print(y, yerr)
    #                 ax[i].plot(xj, y, marker=marker_style, color=color_val, alpha=alpha_val,
    #                            markersize=markersize_val, zorder=zorder_val)
    #                 arrow_len = 0.25
    #                 ax[i].annotate(
    #                     '', xy=(xj, y + arrow_len), xytext=(xj, y),
    #                     arrowprops=dict(
    #                         arrowstyle='-|>',
    #                         color=color_val,
    #                         lw=elw_val*2,
    #                         alpha=alpha_val,
    #                         mutation_scale=18  # 控制箭头头部大小
    #                     ),
    #                     zorder=zorder_val,
    #                     annotation_clip=False
    #                 )
    #             else:
    #                 ax[i].errorbar(xj, y, xerr=xerrj, yerr=yerr,
    #                                fmt=marker_style, color=color_val, alpha=alpha_val,
    #                                markersize=markersize_val, elinewidth=elw_val, 
    #                                capsize=capsize_val, zorder=zorder_val)

    #     ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%dT%H:%M'))
    #     plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=16)
    #     fy_lim = cal_ylim(fy, fy_err)
    #     ax[i].set_ylim(fy_lim[0], fy_lim[1])
    #     ax[i].set_ylabel('Mag')
    #     ax[i].invert_yaxis()
    #     ax[i].grid(True, linestyle='--', alpha=0.6)  # 可选：设置虚线网格，透明度 0.6

    #     # **标注波段**
    #     band_text = f'{target_nm} ({band_lst[i]})'
    #     ax[i].text(0.97, 0.9, band_text, transform=ax[i].transAxes, 
    #                color=f'C{i}', fontsize=30., fontweight='bold', 
    #                fontfamily='serif', horizontalalignment='right')

    # ax[-1].set_xlabel('Obs. Time')
    # # ** 标注 T0 和 r_aper**
    # if t0:
    #     t0_str = t0.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # 只保留毫秒 3 位
    #     t0r_text = r'$T_0$={}, r={}'.format(t0_str, r_lst[0])
    # else:
    #     t0r_text = f'r = {r_lst[0]}'
    # ax[-1].text(0.03, 0.06, t0r_text, transform=ax[-1].transAxes, 
    #             color='k', fontsize=24., fontweight='bold', 
    #             fontfamily='serif', horizontalalignment='left')

    # lc_pdfnm = lc_csvnm.replace('.csv', '.pdf')
    # plt.savefig(lc_pdfnm, format='pdf', dpi=1200, 
    #             orientation='landscape', bbox_inches='tight')
    # # plt.show()
    # plt.close()

    # **画时间减 t0 后 log 的图**
    arrow_len = 0.5  # 极限星等的箭头长度
    if t0:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(16, 10))
        for i in range(len(band_lst)):
            dfb = df[df['band'] == band_lst[i]]
            for is_single in [True, False]:
                dfb_sub = dfb[dfb['ncombine'] == 1] if is_single else dfb[dfb['ncombine'] != 1]
                if dfb_sub.empty:
                    continue
                # 使用 t_start 和 t_end 计算横坐标和误差
                t_start = pd.to_datetime(dfb_sub['t_start'])
                t_end = pd.to_datetime(dfb_sub['t_end'])
                t_mid = t_start + (t_end - t_start) / 2
                t_xerr = (t_end - t_start) / 2
                t_mid_sec = (t_mid - t0).dt.total_seconds()
                t_xerr_sec = t_xerr.dt.total_seconds()
                fy, fy_err = dfb_sub['mag'], dfb_sub['mag_err']
                fy_err = pd.to_numeric(fy_err, errors='coerce')  # 强制转为 float，非数值变 nan
                # **区分两种数据点风格**
                # custom_colors = ['blue', 'red']
                # custom_colors = ['cyan', 'red']
                custom_colors = ['C0', 'C1']
                color_val = f'C{i}' if is_single else custom_colors[i]
                marker_style = 'o' if is_single else 'D'
                alpha_val = 0.4 if is_single else 0.8
                markersize_val = markersize * 0.75 if is_single else markersize * 0.75
                elw_val = elw * 2 if is_single else elw * 2
                capsize_val = elw * 8 if is_single else elw * 8
                label_tag = f'{band_lst[i]}-single' if is_single else f'{band_lst[i]}-stacked'
                zorder_val = 1 if is_single else 3
                # **如果 yerr = 0，则只画上限值，同时画出数据点向下的箭头，其他点则正常画误差棒**
                for j in range(len(fy)):
                    x = t_mid_sec.iloc[j]
                    xerr = t_xerr_sec.iloc[j]
                    y = fy.iloc[j]
                    yerr = fy_err.iloc[j]
                    label_tag_val = label_tag if j == 0 else None  # 只在第一个点加图例
                    if np.isnan(yerr):  # mag_limit点，画marker+箭头
                        ax.errorbar(x, y, xerr=xerr,  # 如果是上限，先画数据点
                                    fmt=marker_style, color=color_val, alpha=alpha_val, 
                                    markersize=markersize_val, elinewidth=elw_val,
                                    capsize=capsize_val, label=label_tag_val, zorder=zorder_val)
                        # mag_limit 横线范围 [t_start-t0, t_end-t0]
                        x0 = (t_start.iloc[j] - t0).total_seconds()
                        x1 = (t_end.iloc[j] - t0).total_seconds()
                        # ax.hlines(y, x0, x1, color=color_val, linestyle='--', alpha=alpha_val, linewidth=elw_val*2, zorder=zorder_val)
                        # ax.arrow(x, y, 0, arrow_len, length_includes_head=True,
                        #          head_width=16 * (x1-x0)/10, head_length=arrow_len / 3,
                        #          fc=color_val, ec=color_val, alpha=alpha_val, zorder=zorder_val)、

                        ax.annotate(
                            '', xy=(x, y + arrow_len), xytext=(x, y),
                            arrowprops=dict(
                                arrowstyle='-|>',
                                color=color_val,
                                lw=elw_val*2,
                                alpha=alpha_val,
                                mutation_scale=18  # 控制箭头头部大小
                            ),
                            zorder=zorder_val,
                            annotation_clip=False
                        )
                    else:
                        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                                    fmt=marker_style, color=color_val, alpha=alpha_val,
                                    markersize=markersize_val, elinewidth=elw_val, 
                                    capsize=capsize_val, label=label_tag_val, zorder=zorder_val)
            ax.legend(loc=[0.03, 0.1], fontsize=16., ncol=len(band_lst), framealpha=0.5)
            # x 轴范围自适应
            all_t0 = (pd.to_datetime(dfb['t_start']) - t0).dt.total_seconds().tolist()
            all_t1 = (pd.to_datetime(dfb['t_end']) - t0).dt.total_seconds().tolist()
        
        # **设置 x 轴和 y 轴范围**
        fx_lim = cal_xloglim(np.array(all_t0 + all_t1))
        ax.set_xlim(fx_lim[0], fx_lim[1])
        fy_lim = cal_ylim(df['mag'], df['mag_err'], nlim0=0.1, nlim1=0.1)
        ax.set_ylim(fy_lim[0], fy_lim[1])

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


def main():
    # **读取参数配置文件**
    config_filenm = sys.argv[1]
    with open(config_filenm, 'r') as f:
        config = yaml.safe_load(f)
    # target_ra, target_dec = config['target_radec']  # 读取目标天球坐标 (RA, Dec)
    target_nm = config['target_nm']  # 目标名称
    raper_chos = config['raper_chos']
    r_cho, r_in, r_out, r_step, r_all = raper_chos
    # print(r_cho, r_in, r_out, r_step, r_all)
    t0 = pd.to_datetime(config['t0'])
    # print(t0)

    out_dir = 'res'
    res_csv = os.path.join(out_dir, f"{target_nm}_VT.csv")

    plot_lc(res_csv, target_nm, t0=t0)

if __name__ == '__main__':
    start_time = time.time()
    
    main()

    # Cal Time
    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')

