#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中合并所有已处理轨次的lc csv，并统一绘图
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd


def filter_outliers(df, sn_col='sn', sn_thresh_B=0.5, sn_thresh_R=1):
    '''
    不同band采用不同sn阈值，VT_B阈值为0.5，VT_R阈值为1
    '''
    if sn_col not in df.columns or 'band' not in df.columns:
        return df
    cond_b = (df['band'] == 'VT_B') & (df[sn_col] >= sn_thresh_B)
    cond_r = (df['band'] == 'VT_R') & (df[sn_col] >= sn_thresh_R)
    return df[cond_b | cond_r].copy()


def main():
    if len(sys.argv) < 3:
        print("用法: merge_all_orbit_lc.py config.yaml update.lst [proc_path] [code_path]")
        sys.exit(1)
    config_file = sys.argv[1]
    update_lst = sys.argv[2]
    proc_path = sys.argv[3] if len(sys.argv) > 3 else "../proc"

    # 读取目标名和孔径
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    target_nm = config['target_nm']
    r_cho = config['raper_chos'][0]
    r_cho_text = f"{r_cho:.1f}".replace('.', '_')
    lc_csvnm = f"{target_nm}_lc_{r_cho_text}.csv"
    # print(r_cho, lc_csvnm)

    # 遍历update.lst收集所有csv
    csv_list = []
    with open(update_lst) as f:
        for line in f:
            orbit_dir = line.strip()
            if not orbit_dir:
                continue
            csv_path = os.path.join(proc_path, orbit_dir, lc_csvnm)
            if os.path.isfile(csv_path):
                csv_list.append(csv_path)
            else:
                print(f"[WARNING] Not Found: {csv_path}")
    if not csv_list:
        print("No lc csv found, exit.")
        sys.exit(1)

    # 合并所有csv，并加入目标名和孔径半径两列
    df_all = pd.read_csv(csv_list[0])
    for csvf in csv_list[1:]:
        df = pd.read_csv(csvf)
        df_all = pd.concat([df_all, df], ignore_index=True)
    # df_all = filter_outliers(df_all)  # 默认使用 sn_thresh_B=0.5, sn_thresh_R=1
    # df_all = filter_outliers(df_all, sn_thresh_B=3, sn_thresh_R=3)  # 两个波段均按 3 sigma 筛选数据
    df_all = filter_outliers(df_all, sn_thresh_B=0, sn_thresh_R=0)
    # 数据保存到 res 目录下
    out_dir = 'res'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_csv = os.path.join(out_dir, lc_csvnm)
    df_all.to_csv(out_csv, index=False)
    print(f"Merged and saved: {out_csv}, total {len(df_all)} rows")


if __name__ == "__main__":
    main()
