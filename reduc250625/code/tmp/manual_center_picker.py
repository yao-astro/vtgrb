import os
import sys
import pandas as pd
import numpy as np
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt

def pick_center_star(img, pos_csvnm, search_radius=20):
    # 读取星表
    df = pd.read_csv(pos_csvnm)
    hdu = pyfits.open(img)
    header = hdu[0].header
    naxis1, naxis2 = int(header['NAXIS1']), int(header['NAXIS2'])
    center = np.array([naxis1 / 2, naxis2 / 2])
    coords = df[['xcent', 'ycent']].values
    dists = np.linalg.norm(coords - center, axis=1)
    # 找到中心附近的星
    close_idx = np.where(dists < search_radius)[0]
    if len(close_idx) == 0:
        print(f'No stars found within {search_radius} pixels of center in {pos_csvnm}')
        return None
    # 画出星表和中心附近的星
    plt.figure(figsize=(8,8))
    plt.imshow(hdu[0].data, cmap='gray', origin='lower', vmin=np.percentile(hdu[0].data,5), vmax=np.percentile(hdu[0].data,99))
    plt.scatter(df['xcent'], df['ycent'], s=20, edgecolor='blue', facecolor='none', label='all stars')
    plt.scatter(df.iloc[close_idx]['xcent'], df.iloc[close_idx]['ycent'], s=60, edgecolor='red', facecolor='none', label='center candidates')
    for i in close_idx:
        row = df.iloc[i]
        plt.text(row['xcent']+2, row['ycent']+2, str(int(row['obj_id'])), color='red', fontsize=12)
    plt.scatter([center[0]], [center[1]], marker='+', color='yellow', s=100, label='center')
    plt.legend()
    plt.title(f"{os.path.basename(img)}: Pick center star obj_id")
    plt.show()
    # 人工输入
    print(f"Candidates obj_id: {df.iloc[close_idx]['obj_id'].tolist()}")
    obj_id = input("请输入你认为的中心星 obj_id：")
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

def main():
    # 用法: python manual_center_picker.py fit_list.txt
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
        obj_id = pick_center_star(img, pos_csvnm)
        if obj_id is not None:
            results.append({'img': os.path.basename(img), 'obj_id': obj_id})
    # 保存人工选择结果
    if results:
        df_choice = pd.DataFrame(results)
        df_choice.to_csv('center_choice.csv', index=False)
        print("人工选择结果已保存到 center_choice.csv")

if __name__ == "__main__":
    main()