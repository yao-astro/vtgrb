#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
以指定日期（如2024-11-19）为基准，计算所有观测点的相对星等及误差
输入：all_orbit_lc.csv（或stat csv），输出：all_orbit_lc_rel.csv
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def mag_rel_by_date(csv_file, base_date="2024-11-19", out_csv=None, date_col="t_center", mag_col="mag_median", err_col="mag_std", band_col="band"):
    df = pd.read_csv(csv_file)
    # 只保留与基准日同一天的点为基准
    base_rows = df[df[date_col].str.startswith(base_date)]
    if base_rows.empty:
        print(f"[ERROR] 未找到基准日期 {base_date} 的观测点！")
        sys.exit(1)
    # 按band分别作差
    rel_mag = []
    for band in df[band_col].unique():
        df_band = df[df[band_col]==band].copy()
        base = base_rows[base_rows[band_col]==band]
        if base.empty:
            print(f"[WARN] 基准日无 {band} 波段，跳过")
            continue
        base_mag = base[mag_col].values[0]
        base_err = base[err_col].values[0]
        df_band['mag_rel'] = df_band[mag_col] - base_mag
        df_band['mag_rel_err'] = np.sqrt(df_band[err_col]**2 + base_err**2)
        rel_mag.append(df_band)
    if not rel_mag:
        print("[ERROR] 没有可用的基准波段数据！")
        sys.exit(1)
    df_rel = pd.concat(rel_mag, ignore_index=True)
    if out_csv is None:
        out_csv = csv_file.replace('.csv', '_rel.csv')
    df_rel.to_csv(out_csv, index=False)
    print(f"已保存相对星等结果: {out_csv}")
    return out_csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: mag_rel_by_date.py all_orbit_lc.csv [基准日期:2024-11-19]")
        sys.exit(1)
    csv_file = sys.argv[1]
    base_date = sys.argv[2] if len(sys.argv) > 2 else "2024-11-19"
    mag_rel_by_date(csv_file, base_date)
