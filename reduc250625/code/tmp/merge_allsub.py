#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中合并所有已处理轨次的lc csv，并统一绘图
"""
import os
import sys
import yaml
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("用法: merge_all_orbit_lc.py config.yaml update.lst [proc_path] [code_path]")
        sys.exit(1)
    config_file = sys.argv[1]
    update_lst = sys.argv[2]
    proc_path = sys.argv[3] if len(sys.argv) > 3 else "../proc"
    code_path = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(os.path.abspath(__file__))

    # 读取目标名和孔径
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    target_nm = config['target_nm']
    r_cho = config['raper_chos'][0]
    r_cho_text = f"{r_cho:.1f}".replace('.', '_')
    lc_csvnm = f"{target_nm}_lc_{r_cho_text}_stat.csv"

    # 遍历update.lst收集所有csv
    csv_list = []
    with open(update_lst) as f:
        for line in f:
            orbit_dir = line.strip()
            if not orbit_dir:
                continue
            csv_path = os.path.join(proc_path, orbit_dir, 'lc', lc_csvnm)
            if os.path.isfile(csv_path):
                csv_list.append(csv_path)
            else:
                print(f"[WARN] 未找到: {csv_path}")
    if not csv_list:
        print("未找到任何轨次lc csv，退出")
        sys.exit(1)

    # 合并所有csv，并加入目标名和孔径半径两列
    df_all = pd.read_csv(csv_list[0])
    df_all['target_nm'] = target_nm
    df_all['r_aper'] = r_cho
    for csvf in csv_list[1:]:
        df = pd.read_csv(csvf)
        df['target_nm'] = target_nm
        df['r_aper'] = r_cho
        df_all = pd.concat([df_all, df], ignore_index=True)
    out_dir = 'res'
    out_csv = os.path.join(out_dir, "all_orbit_lc.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"已合并保存: {out_csv}，共{len(df_all)}行")


if __name__ == "__main__":
    main()
