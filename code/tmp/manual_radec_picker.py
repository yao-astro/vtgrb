import os
import sys
import pandas as pd
import numpy as np
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import random

def pick_obj_by_radec(img, pos_csvnm, search_radius=3):  # 单位: 角秒
    df = pd.read_csv(pos_csvnm)
    hdu = pyfits.open(img)
    header = hdu[0].header
    obj_ra = float(header.get('OBJ_RA', 0))
    obj_dec = float(header.get('OBJ_DEC', 0))
    c_obj = SkyCoord(ra=obj_ra, dec=obj_dec, unit='deg')
    c_stars = SkyCoord(ra=df['ra'].values, dec=df['dec'].values, unit='deg')
    dists = c_stars.separation(c_obj).arcsec
    df['dist_to_obj'] = dists
    close_idx = np.where(dists < search_radius)[0]
    if len(close_idx) == 0:
        print(f'No stars found within {search_radius} arcsec of OBJ_RA/OBJ_DEC in {pos_csvnm}')
        return None
    # 画图
    plt.figure(figsize=(8,8))
    plt.imshow(hdu[0].data, cmap='gray', origin='lower', vmin=np.percentile(hdu[0].data,5), vmax=np.percentile(hdu[0].data,99))
    plt.scatter(df['xcent'], df['ycent'], s=20, edgecolor='blue', facecolor='none', label='all stars')
    plt.scatter(df.iloc[close_idx]['xcent'], df.iloc[close_idx]['ycent'], s=60, edgecolor='red', facecolor='none', label='candidates')
    for i in close_idx:
        row = df.iloc[i]
        plt.text(row['xcent']+2, row['ycent']+2, f"{int(row['obj_id'])}\n{row['flux']:.1f}", color='red', fontsize=10)
    # 画出目标天球坐标对应的像素位置
    from astropy.wcs import WCS
    wcs = WCS(header)
    x_obj, y_obj = wcs.all_world2pix([[obj_ra, obj_dec]], 1)[0]
    plt.scatter([x_obj], [y_obj], marker='+', color='yellow', s=100, label='OBJ_RA/OBJ_DEC')
    plt.legend()
    plt.title(f"{os.path.basename(img)}: Pick target obj_id")
    plt.show()
    print(f"Candidates obj_id: {df.iloc[close_idx]['obj_id'].tolist()}")
    obj_id = input("请输入你认为的目标 obj_id：")
    try:
        obj_id = int(obj_id)
        if obj_id in df.iloc[close_idx]['obj_id'].values:
            return obj_id
        else:
            print("输入的 obj_id 不在候选列表中。")
            return None
    except Exception as e:
        print("输入无效。")
        return None

def copy_random_suc_to_ml(proc_date_dir, ml_dir, n=10):
    suc_path = os.path.join(proc_date_dir, 'suc.lst')
    if not os.path.exists(suc_path):
        print(f"{suc_path} not found.")
        return
    with open(suc_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) == 0:
        print(f"{suc_path} is empty.")
        return
    sample_n = min(n, len(lines))
    sampled = random.sample(lines, sample_n)
    os.makedirs(ml_dir, exist_ok=True)
    ml_lst_path = os.path.join(ml_dir, 'ml.lst')
    with open(ml_lst_path, 'w') as f:
        for line in sampled:
            f.write(line + '\n')
    print(f"已从{suc_path}随机抽取{sample_n}行，保存到{ml_lst_path}")

def main():
    # 用法: python manual_obj_picker.py fit_list.txt 或 python manual_radec_picker.py --sample proc/date ml
    if len(sys.argv) > 2 and sys.argv[1] == '--sample':
        proc_date_dir = sys.argv[2]
        ml_dir = sys.argv[3] if len(sys.argv) > 3 else 'ml'
        copy_random_suc_to_ml(proc_date_dir, ml_dir)
        return
    fit_lstnm = sys.argv[1]
    fit_lst = np.loadtxt(fit_lstnm, dtype=str)
    fit_lst = np.atleast_1d(fit_lst).tolist()
    fit_lst.sort()
    imgs = [i.strip() for i in fit_lst]
    results = []
    for img in imgs:
        pos_csvnm = os.path.splitext(os.path.basename(img))[0] + '_pos.csv'
        if not os.path.exists(pos_csvnm):
            print(f"{pos_csvnm} not found, skip.")
            continue
        obj_id = pick_obj_by_radec(img, pos_csvnm)
        if obj_id is not None:
            results.append({'img': os.path.basename(img), 'obj_id': obj_id})
    if results:
        df_choice = pd.DataFrame(results)
        df_choice.to_csv('obj_choice.csv', index=False)
        print("人工选择结果已保存到 obj_choice.csv")

if __name__ == "__main__":
    main()