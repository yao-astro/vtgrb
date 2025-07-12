import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize
from tqdm import tqdm

# 交互式人工选择星中心的类
class ManualSelector:
    def __init__(self, img, x_guess, y_guess, r_cho, title=None):
        """
        img: 二维图像数据（numpy数组）
        x_guess, y_guess: 自动匹配得到的初始xy坐标（在subimg中的坐标）
        r_cho: 圈的像素半径
        title: 窗口标题
        """
        self.img = img
        self.x = None  # 用户点击选中的x坐标
        self.y = None  # 用户点击选中的y坐标
        self.x_guess = x_guess  # 自动匹配的x
        self.y_guess = y_guess  # 自动匹配的y
        self.r_cho = r_cho
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=SqrtStretch())
        self.ax.imshow(img, cmap='gray', origin='lower', norm=norm)
        # 计算marker大小，使圈半径为r_cho像素
        bbox = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        pix2pt = (bbox.width * 72) / img.shape[1]
        self.s_circle = (r_cho * pix2pt) ** 2
        # 用红圈标注自动匹配位置
        self.ax.scatter([x_guess], [y_guess], s=self.s_circle, edgecolor='red', facecolor='none', marker='o', label='Auto', zorder=10)
        self.ax.set_title(title or 'Manual Select')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.selected = False
        # self.ax.legend()
        plt.xlabel('X [pix]')
        plt.ylabel('Y [pix]')
        plt.tight_layout()

    def onclick(self, event):
        """
        鼠标点击事件回调，记录用户点击的坐标，并用蓝色圈标记，等待用户回车确认
        """
        if event.inaxes != self.ax:
            return
        self.x, self.y = event.xdata, event.ydata
        if hasattr(self, 'manual_marker') and self.manual_marker is not None:
            self.manual_marker.remove()
        # 用蓝色圈标记人工选择点，半径与r_cho一致
        self.manual_marker = self.ax.scatter([self.x], [self.y], s=self.s_circle, edgecolor='blue', facecolor='none', marker='o', label='Manual', zorder=11)
        self.ax.legend()
        self.fig.canvas.draw()
        # 不关闭窗口，等待回车

    def onkey(self, event):
        """
        键盘事件回调，按回车确认选择，关闭窗口
        """
        if event.key == 'enter' and self.x is not None and self.y is not None:
            self.selected = True
            plt.close(self.fig)

    def get(self):
        """
        显示窗口，等待用户点击和回车，返回选中的xy坐标
        """
        self.manual_marker = None
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        plt.show()
        return self.x, self.y


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
    # t0 = pd.to_datetime(config['t0'])
    # print(t0)
    # rlst = np.round([r_cho, r_all], 1).tolist()  # 孔径测光半径 list
    # rlst = np.round(np.arange(r_step, r_in + r_step, r_step), 1).tolist()  # 计算孔径半径列表
    # print(rlst)

    # 读取fit列表文件名
    fit_lstnm = sys.argv[2]
    if not os.path.exists(fit_lstnm):
        print(f'Error: {fit_lstnm} does not exist')
        sys.exit(1)
    # 读取fit文件列表
    fit_lst = np.loadtxt(fit_lstnm, dtype=str)
    fit_lst = np.atleast_1d(fit_lst).tolist()
    fit_lst.sort()
    imgs = [i.strip() for i in fit_lst]
    if not imgs:
        print(f'Error: {fit_lstnm} is empty')
        sys.exit(1)
    
    pos_dir = 'pos'  # 星表目录
    if not os.path.exists(pos_dir):
        print('No [pos] Dir, back to Step 3. Finding Stars')
        sys.exit(1)
    
    results = []  # 存储每帧人工/自动选定的结果
    for k in tqdm(range(len(imgs)), desc='Manual GRB Matching'):
        try:
            # 读取FITS头，获取WCS和GRB天球坐标
            with pyfits.open(imgs[k]) as hdul:
                header = hdul[0].header
                obj_ra = float(header.get('OBJ_RA', 0))
                obj_dec = float(header.get('OBJ_DEC', 0))
                wcs = WCS(header)
                grb_x, grb_y = wcs.all_world2pix(obj_ra, obj_dec, 1)
        except Exception as e:
            print(f'Error reading FITS header or WCS for {imgs[k]}: {e}')
            continue
        # 自动选取目标附近的亮星
        pos_csvnm = os.path.splitext(os.path.basename(imgs[k]))[0] + '_pos.csv'
        pos_csv = os.path.join(pos_dir, pos_csvnm)
        if not os.path.exists(pos_csv):
            print(f'Warning: {pos_csv} does not exist')
            continue
        df = pd.read_csv(pos_csv)
        if len(df) == 0:
            print(f'Warning: {pos_csv} 星表为空')
            continue
        # 计算所有星与GRB理论像素的距离，选最近的N颗亮星
        df['dist_pix'] = np.sqrt((df['xcent'] - grb_x)**2 + (df['ycent'] - grb_y)**2)
        search_radius = 500  # 匹配半径（像素）
        flux_thresh = np.percentile(df['flux'], 70)  # 取前30%最亮的星
        df_bright = df[(df['flux'] >= flux_thresh)]
        df_near = df_bright[df_bright['dist_pix'] < search_radius]
        ref_stars = df_near.nsmallest(20, 'dist_pix')  # 选20颗最近亮星
        # 显示自动定位结果，人工选择
        with pyfits.open(imgs[k]) as hdul:
            img = hdul[0].data
        zoom_size = 30  # 放大窗口大小（像素）
        x0, y0 = grb_x - 1, grb_y - 1  # FITS像素从1开始，Python从0
        x1, x2 = int(x0 - zoom_size), int(x0 + zoom_size)
        y1, y2 = int(y0 - zoom_size), int(y0 + zoom_size)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img.shape[1])
        y2 = min(y2, img.shape[0])
        subimg = img[y1:y2, x1:x2]  # 截取局部小图
        # 交互式人工选择GRB中心
        selector = ManualSelector(subimg, grb_x-1-x1, grb_y-1-y1, r_cho * 2, title=os.path.basename(imgs[k]))
        x_sel, y_sel = selector.get()
        # 保存图片，包含自动和人工两个圈
        outdir = 'grb_match_plots'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig, ax = plt.subplots(figsize=(8, 8))
        norm = ImageNormalize(subimg, interval=ZScaleInterval(), stretch=SqrtStretch())
        ax.imshow(subimg, cmap='gray', origin='lower', norm=norm)
        # 计算marker大小，使圈半径为r_cho像素
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        pix2pt = (bbox.width * 72) / subimg.shape[1]  # 1像素对应多少points
        s_circle = (r_cho * 2 * pix2pt) ** 2
        # # 辅助验证：画一条长度为r_cho像素的红色线段
        # ax.plot([grb_x-1-x1, grb_x-1-x1 + r_cho], [grb_y-1-y1, grb_y-1-y1], color='red', lw=2, label='r_cho length')
        # 自动识别圈（红色）
        ax.scatter([grb_x-1-x1], [grb_y-1-y1], s=s_circle, edgecolor='red', facecolor='none', marker='o', label='Auto', zorder=10)
        ax.text(grb_x-x1+0.5, grb_y-y1+0.5, f"({grb_x:.4f},{grb_y:.4f})", color='red', fontsize=9, zorder=11)
        # 人工圈（蓝色）
        if x_sel is not None and y_sel is not None:
            ax.scatter([x_sel], [y_sel], s=s_circle, edgecolor='blue', facecolor='none', marker='o', label='Manual', zorder=12)
            ax.text(x_sel+1.5, y_sel+1.5, f"({x_sel+x1+1:.4f},{y_sel+y1+1:.4f})", color='blue', fontsize=9, zorder=13)
        ax.set_title(os.path.basename(imgs[k]), fontsize=10)
        ax.set_xlabel('X [pix]')
        ax.set_ylabel('Y [pix]')
        # ax.legend()
        plt.tight_layout()
        outpng = os.path.join(outdir, os.path.splitext(os.path.basename(imgs[k]))[0] + '_grb_match.png')
        plt.savefig(outpng, dpi=120)
        plt.close(fig)
        if x_sel is not None and y_sel is not None:
            # 用户点击后，坐标转回原图像素坐标（FITS从1开始）
            x_final = x_sel + x1 + 1
            y_final = y_sel + y1 + 1
            results.append({'frame': imgs[k], 'xcent': x_final, 'ycent': y_final, 'method': 'manual'})
        else:
            # 未点击则用自动匹配结果
            results.append({'frame': imgs[k], 'xcent': grb_x, 'ycent': grb_y, 'method': 'auto'})
    # 保存所有帧的人工/自动选定结果到csv
    df_out = pd.DataFrame(results)
    df_out.to_csv('grb_xyposs.csv', index=False)
    print('Saved manual GRB pixel coordinates for each frame to [grb_xyposs.csv]')

if __name__ == '__main__':
    main()
