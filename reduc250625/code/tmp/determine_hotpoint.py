import os
import sys
import glob
import time
import pandas as pd
import numpy as np


def plot_stars(img, pos_csvnm, id_colnm='obj_id'):
    '''
    使用 DS9 圈出所找到的星
    '''
    import pyds9
    # print(pyds9.ds9_targets())
    d = pyds9.DS9()  # will open a new ds9 window or connect to an existing one
    d.set('width 2000')
    d.set('height 2000')
    d.set(f'file {img}')  # send the file to the open ds9 session
    d.set('zoom to fit')  # 适应窗口大小
    d.set('scale zscale')  # 设置 scale 为 zscale
    r_aper = 3
    df = pd.read_csv(pos_csvnm)
    for index, row in df.iterrows():  # 遍历筛选后的数据并打印 regions add circle 命令
        nid = row[id_colnm]
        nxi = row['xcent']
        nyi = row['ycent']
        is_star = row['is_star']
        if is_star == 0:
            reg = f'image;circle({nxi},{nyi},{r_aper}) # width=1 color=red text={{{nid}}}'  # example region
            d.set('regions', reg)  # load that region in ds9
        else:
            reg = f'image;circle({nxi},{nyi},{r_aper}) # width=2 text={{{nid}}}'  # example region
        #     # reg += f'\nimage;point({nxi},{nyi}) # point=cross {30} width=2 color=red'  # example region
        d.set('regions', reg)  # load that region in ds9
    png_nm = os.path.splitext(os.path.abspath(pos_csvnm))[0] + '.png'
    # d.set(f'export png {png_nm}')  # 以高质量的方式导出 DS9 显示的内容 （只能导出一幅图）
    # print(pos_csvnm, png_nm)
    d.set(f'saveimage png {png_nm}')


def main():
    start_time = time.time()

    # **检查并读取 fit list**
    fit_lstnm = sys.argv[1]
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

    # 读取所有星表（只处理本次fit list对应的csv）
    csvs = [os.path.splitext(os.path.basename(f))[0] + '_pos.csv' for f in imgs]
    # all_stars = []
    # for fname in csvs:
    #     if not os.path.exists(fname):
    #         continue
    #     df = pd.read_csv(fname)
    #     df['file'] = fname
    #     all_stars.append(df)
    # all_stars = pd.concat(all_stars, ignore_index=True)

    # # 构建KDTree用于快速查找邻居
    # from scipy.spatial import cKDTree
    # coords = all_stars[['xcent', 'ycent']].values
    # tree = cKDTree(coords)

    # # 对每个星点，统计其它帧中是否有邻居（3像素内，且不是同一帧）
    # is_star = []
    # for idx, row in all_stars.iterrows():
    #     idxs = tree.query_ball_point([row['xcent'], row['ycent']], r=3)
    #     files = set(all_stars.iloc[i]['file'] for i in idxs if all_stars.iloc[i]['file'] != row['file'])
    #     is_star.append(1 if len(files) >= 1 else 0)
    # all_stars['is_star'] = is_star

    # # 写回每个csv
    # for fname in csvs:
    #     df = all_stars[all_stars['file'] == fname].copy()
    #     df.drop(columns=['file'], inplace=True)
    #     df.to_csv(fname.replace('_pos.csv', '_pos2.csv'), index=False)
    #     print(f'已处理: {fname} -> {fname.replace('_pos.csv', '_pos2.csv')}')

    # 绘图
    for img, csv in zip(imgs, csvs):
        if not os.path.exists(csv):
            print(f'Warning: {csv} does not exist')
            continue
        plot_stars(img, csv.replace('_pos.csv', '_pos2.csv'), id_colnm='obj_id')

    print('全部完成！')

    end_time = time.time()
    run_time = end_time - start_time
    current_file = os.path.basename(__file__)
    print(f'Run Time of [{current_file}]: {run_time:.2f} s')

if __name__ == '__main__':
    main()