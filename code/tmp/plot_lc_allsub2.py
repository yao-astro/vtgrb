#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立脚本：根据合并/统计csv绘制html光变曲线
用法: plot_lc_allsub.py stat_csv [target_nm]
"""
import os
import sys
import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.append('/home/yao/astroWORK/codesyao/plot')
try:
    from pyplotsettings import set_mpl_style
    set_mpl_style()
    markersize, elw = 12, 0.5
except ImportError:
    print('警告：未找到pyplotsettings.py，部分matplotlib美化设置可能无效')


def plot_lc_rel2_html(stat_csv, target_nm=None):
    '''
    根据相对星等csv（band, t_center, t_span_sec, mag_rel, mag_rel_err, target_nm）绘制html光变曲线，区分不同目标源
    '''
    df = pd.read_csv(stat_csv)
    if 'target_nm' not in df.columns:
        # 兼容无target_nm列的情况
        df['target_nm'] = target_nm if target_nm is not None else 'Target'
    targets = df['target_nm'].unique()
    markersize, elw = 18, 1.2
    color_map = ['cyan', 'orange', 'magenta', 'yellow', 'green', 'blue', 'red', 'lime', 'purple', 'brown']
    fig = go.Figure()
    for ti, tgt in enumerate(targets):
        dft = df[df['target_nm'] == tgt]
        band_lst = dft['band'].unique()
        for i, band in enumerate(band_lst):
            dfg = dft[dft['band'] == band]
            t_center = pd.to_datetime(dfg['t_center'])
            mag_rel = dfg['mag_rel']
            mag_rel_err = dfg['mag_rel_err']
            t_span = dfg['t_span_sec']
            color = color_map[(ti*5 + i) % len(color_map)]
            fig.add_trace(
                go.Scatter(
                    x=t_center,
                    y=mag_rel,
                    error_y=dict(type='data', array=mag_rel_err, visible=True, thickness=elw),
                    error_x=dict(type='data', array=t_span, visible=True, thickness=elw),
                    mode='markers',
                    marker=dict(color=color, size=markersize / 1.2, symbol='circle'),
                    name=f'{tgt}-{band}',
                    showlegend=True,
                    hovertemplate=f"{tgt}-{band}<br>mag_rel=%{{y:.3f}}±%{{customdata[0]:.3f}}<br>time=%{{x|%Y-%m-%dT%H:%M:%S}}±%{{customdata[1]:.0f}}s<extra></extra>",
                    customdata=np.stack([mag_rel_err, t_span], axis=-1)
                )
            )
    fig.update_yaxes(autorange="reversed")
    fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, x1=1, y0=0, y1=0,
        line=dict(color="white", width=3, dash="dash"),
        layer="below"
    )
    fig.update_layout(
        title=dict(
            text=f"<b>Relative Light Curve (BD+28D4211 & Feige34)</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24)
        ),
        xaxis=dict(title="Obs. Time", tickformat="%y-%m-%d", showgrid=True, tickangle=45, tickfont=dict(size=14)),
        yaxis=dict(title="Mag_rel (VS. 2024-11-19)", showgrid=True),
        template="plotly_dark",
        width=1200,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    htmlnm = stat_csv.replace('.csv', '.html')
    fig.write_html(htmlnm)
    print(f"Relative stat light curve plot saved to {htmlnm}")


def plot_lc_rel2_pdf(stat_csv, target_nm=None):
    '''
    根据相对星等csv（band, t_center, t_span_sec, mag_rel, mag_rel_err, target_nm）绘制pdf光变曲线，区分不同目标源
    主图添加指定放大区域方框，并在主图内部右中位置画该区域的放大图
    '''
    df = pd.read_csv(stat_csv)
    if 'target_nm' not in df.columns:
        df['target_nm'] = target_nm if target_nm is not None else 'Target'
    targets = df['target_nm'].unique()
    color_map = ['cyan', 'orange', 'magenta', 'yellow', 'green', 'blue', 'red', 'lime', 'purple', 'brown']
    x0, x1 = pd.to_datetime('2024-11-15'), pd.to_datetime('2025-01-01')
    y0, y1 = 0.06, -0.07
    fig, ax = plt.subplots(figsize=(16, 12))
    for ti, tgt in enumerate(targets):
        dft = df[df['target_nm'] == tgt]
        band_lst = dft['band'].unique()
        for i, band in enumerate(band_lst):
            dfg = dft[dft['band'] == band]
            t_center = pd.to_datetime(dfg['t_center'])
            mag_rel = dfg['mag_rel'].values
            mag_rel_err = dfg['mag_rel_err'].values
            t_span = dfg['t_span_sec'].values / 86400
            color = color_map[(ti*5 + i) % len(color_map)]
            t_center_num = mdates.date2num(t_center)
            ax.errorbar(
                t_center_num, mag_rel, yerr=mag_rel_err, xerr=t_span,
                fmt='o', color=color, label=f'{tgt}-{band}', capsize=3, 
                markersize=markersize, elinewidth=elw, alpha=0.85
            )
    ax.axhline(0, color='gray', linestyle='--', linewidth=2)
    # ax.set_title('Relative Light Curve (BD+28D4211 & Feige34)')
    ax.set_xlabel('Obs. Time')
    ax.set_ylabel('Mag_rel (VS. 2024-11-19)')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # 图例设置为一横排，字体更大，位置用数字微调
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=16, ncol=len(by_label), bbox_to_anchor=(0.98, 0.99))
    # 方框
    rect = plt.Rectangle((mdates.date2num(x0), y1), mdates.date2num(x1)-mdates.date2num(x0), y0-y1,
                        linewidth=2, edgecolor='grey', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    # 放大图（主图内部右中）
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axins = inset_axes(ax, width="38%", height="48%", loc='upper right', borderpad=2)
    for ti, tgt in enumerate(targets):
        dft = df[df['target_nm'] == tgt]
        band_lst = dft['band'].unique()
        for i, band in enumerate(band_lst):
            dfg = dft[dft['band'] == band]
            t_center = pd.to_datetime(dfg['t_center'])
            mag_rel = dfg['mag_rel'].values
            mag_rel_err = dfg['mag_rel_err'].values
            t_span = dfg['t_span_sec'].values / 86400
            color = color_map[(ti*5 + i) % len(color_map)]
            t_center_num = mdates.date2num(t_center)
            mask = (t_center >= x0) & (t_center <= x1) & (mag_rel <= y0) & (mag_rel >= y1)
            axins.errorbar(
                t_center_num[mask], mag_rel[mask], yerr=mag_rel_err[mask], xerr=t_span[mask],
                fmt='o', color=color, capsize=3, markersize=markersize, elinewidth=elw, alpha=0.85
            )
    axins.axhline(0, color='gray', linestyle='--', linewidth=2)
    # 设置zoom子图横轴主刻度和显示范围，避免日期混乱
    import matplotlib.ticker as ticker
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    # 使用固定间隔的日期刻度
    import datetime
    total_days = (x1 - x0).days
    if total_days > 20:
        interval = 10  # 每7天一个刻度
    elif total_days > 10:
        interval = 5
    else:
        interval = 1
    major_locator = mdates.DayLocator(interval=interval)
    axins.xaxis.set_major_locator(major_locator)
    axins.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.setp(axins.get_xticklabels(), rotation=45, fontsize=14)  # 设置zoom图横轴刻度字体大小
    plt.setp(axins.get_yticklabels(), fontsize=14)  # 设置zoom图纵轴刻度字体大小
    axins.grid(True, linestyle=':', alpha=0.5)
    # axins.set_title('Zoomed', fontsize=13)
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red", lw=1.5, linestyle='--')
    # 自定义连线：方框右上/右下 -> 子图左上/左下
    from matplotlib.patches import ConnectionPatch
    # 方框右上、右下（主图坐标）
    rect_x = mdates.date2num(x1)
    rect_y_top = y0
    rect_y_bot = y1
    # 子图左上、左下（放大图坐标）
    axins_x = mdates.date2num(x0)
    axins_y_top = y0
    axins_y_bot = y1
    # 右上->左上
    con1 = ConnectionPatch(xyA=(axins_x, axins_y_top), xyB=(rect_x, rect_y_top),
                          coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax,
                          color="red", lw=1.5, linestyle='--', zorder=10)
    # 右下->左下
    con2 = ConnectionPatch(xyA=(axins_x, axins_y_bot), xyB=(rect_x, rect_y_bot),
                          coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax,
                          color="red", lw=1.5, linestyle='--', zorder=10)
    fig.add_artist(con1)
    fig.add_artist(con2)
    # 绘制箭头：整体向右向上平移
    rect_x = mdates.date2num(x1)
    rect_y_mid = (y0 + y1) / 2
    # 箭头起点（主图坐标，方框右中点）
    start_xy = (rect_x, rect_y_mid)
    # 箭头终点（子图坐标，zoom子图的中心点）
    axins_xlim = axins.get_xlim()
    axins_ylim = axins.get_ylim()
    end_xy = ((axins_xlim[0] + axins_xlim[1]) / 2, (axins_ylim[0] + axins_ylim[1]) / 2)
    # 坐标变换到figure
    trans_fig = fig.transFigure.inverted()
    start_disp = ax.transData.transform(start_xy)
    end_disp = axins.transData.transform(end_xy)
    # 平移量（以figure fraction为单位，右上方向）
    shift_x, shift_y = 0.04, 0.12  # 向右0.04，向上0.12
    start_fig = trans_fig.transform(start_disp) + np.array([shift_x, shift_y])
    end_fig = trans_fig.transform(end_disp) + np.array([shift_x, shift_y])
    # 绘制箭头
    plt.annotate('', xy=end_fig, xytext=start_fig, xycoords='figure fraction',
                 textcoords='figure fraction', arrowprops=dict(arrowstyle='fancy', color='grey', lw=elw))
    plt.tight_layout()
    pdfnm = stat_csv.replace('.csv', '.pdf')
    plt.savefig(pdfnm)
    print(f"Relative stat light curve PDF saved to {pdfnm}")
    plt.close()


def plot_lc_rel2_pdf_nozoom(stat_csv, target_nm=None):
    '''
    根据相对星等csv绘制无zoom局部放大图的PDF主图（仅主图，无放大子图和连线）
    '''
    df = pd.read_csv(stat_csv)
    if 'target_nm' not in df.columns:
        df['target_nm'] = target_nm if target_nm is not None else 'Target'
    targets = df['target_nm'].unique()
    color_map = ['cyan', 'orange', 'magenta', 'yellow', 'green', 'blue', 'red', 'lime', 'purple', 'brown']
    x0, x1 = pd.to_datetime('2024-11-15'), pd.to_datetime('2025-01-01')
    y0, y1 = 0.06, -0.07
    fig, ax = plt.subplots(figsize=(16, 12))
    for ti, tgt in enumerate(targets):
        dft = df[df['target_nm'] == tgt]
        band_lst = dft['band'].unique()
        for i, band in enumerate(band_lst):
            dfg = dft[dft['band'] == band]
            t_center = pd.to_datetime(dfg['t_center'])
            mag_rel = dfg['mag_rel'].values
            mag_rel_err = dfg['mag_rel_err'].values
            t_span = dfg['t_span_sec'].values / 86400
            color = color_map[(ti*5 + i) % len(color_map)]
            t_center_num = mdates.date2num(t_center)
            ax.errorbar(
                t_center_num, mag_rel, yerr=mag_rel_err, xerr=t_span,
                fmt='o', color=color, label=f'{tgt}-{band}', capsize=3, 
                markersize=markersize, elinewidth=elw, alpha=0.85
            )
    ax.axhline(0, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('Obs. Time')
    ax.set_ylabel('Mag_rel (VS. 2024-11-19)')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=16, ncol=len(by_label), bbox_to_anchor=(0.98, 0.99))
    # # 方框（可选，主图右上角标记zoom区域，但不画zoom子图）
    # rect = plt.Rectangle((mdates.date2num(x0), y1), mdates.date2num(x1)-mdates.date2num(x0), y0-y1,
    #                     linewidth=2, edgecolor='grey', facecolor='none', linestyle='--')
    # ax.add_patch(rect)
    plt.tight_layout()
    pdfnm = stat_csv.replace('.csv', '_nozoom.pdf')
    plt.savefig(pdfnm)
    print(f"Relative stat light curve PDF (no zoom) saved to {pdfnm}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("用法: plot_lc_allsub2.py stat_csv [target_nm]")
        sys.exit(1)
    stat_csv = sys.argv[1]
    target_nm = sys.argv[2] if len(sys.argv) > 2 else None
    plot_lc_rel2_html(stat_csv, target_nm)
    plot_lc_rel2_pdf(stat_csv, target_nm)
    plot_lc_rel2_pdf_nozoom(stat_csv, target_nm)

if __name__ == "__main__":
    main()
