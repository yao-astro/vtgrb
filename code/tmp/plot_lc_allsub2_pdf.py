#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立脚本：根据合并/统计csv绘制pdf光变曲线，区分不同目标源
用法: plot_lc_allsub2_pdf.py stat_csv [target_nm]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_lc_rel2_pdf(stat_csv, target_nm=None):
    '''
    根据相对星等csv（band, t_center, t_span_sec, mag_rel, mag_rel_err, target_nm）绘制pdf光变曲线，区分不同目标源
    '''
    df = pd.read_csv(stat_csv)
    if 'target_nm' not in df.columns:
        df['target_nm'] = target_nm if target_nm is not None else 'Target'
    targets = df['target_nm'].unique()
    color_map = ['cyan', 'orange', 'magenta', 'yellow', 'green', 'blue', 'red', 'lime', 'purple', 'brown']
    fig, ax = plt.subplots(figsize=(14, 7))
    for ti, tgt in enumerate(targets):
        dft = df[df['target_nm'] == tgt]
        band_lst = dft['band'].unique()
        for i, band in enumerate(band_lst):
            dfg = dft[dft['band'] == band]
            t_center = pd.to_datetime(dfg['t_center'])
            mag_rel = dfg['mag_rel']
            mag_rel_err = dfg['mag_rel_err']
            t_span = dfg['t_span_sec'] / 86400  # 秒转为天
            color = color_map[(ti*5 + i) % len(color_map)]
            ax.errorbar(
                t_center, mag_rel, yerr=mag_rel_err, xerr=t_span,
                fmt='o', color=color, label=f'{tgt}-{band}', capsize=3, markersize=6, alpha=0.85
            )
    ax.axhline(0, color='gray', linestyle='--', linewidth=2)
    ax.set_title('Relative Light Curve (all targets)', fontsize=20)
    ax.set_xlabel('Obs. Time', fontsize=16)
    ax.set_ylabel('Mag_rel (VS. 2024-11-19)', fontsize=16)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.xticks(rotation=45)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=12, ncol=2)
    plt.tight_layout()
    pdfnm = stat_csv.replace('.csv', '.pdf')
    plt.savefig(pdfnm)
    print(f"Relative stat light curve PDF saved to {pdfnm}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("用法: plot_lc_allsub2_pdf.py stat_csv [target_nm]")
        sys.exit(1)
    stat_csv = sys.argv[1]
    target_nm = sys.argv[2] if len(sys.argv) > 2 else None
    plot_lc_rel2_pdf(stat_csv, target_nm)

if __name__ == "__main__":
    main()
