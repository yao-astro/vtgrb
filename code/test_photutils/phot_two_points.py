import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# 用法: python phot_two_points.py img.fits x1 y1 x2 y2
# x1, y1, x2, y2 都是 FITS 原像素坐标（从1开始）
# 测光半径1.5像素，背景环10-15像素

def main():
    # if len(sys.argv) != 6:
    #     print('用法: python phot_two_points.py img.fits x1 y1 x2 y2')
    #     sys.exit(1)
    # fitnm = sys.argv[1]
    # x1, y1, x2, y2 = map(float, sys.argv[2:])
    fitnm = 'SVT_ToO-NOM-GRB_02842_R_02185_241116T192856_46130_c_n68.fit'
    x1, y1 = 1023.99800796812, 1016.91097149862  # 手动更新的坐标
    x2, y2 = 1022.9917601126945, 1014.8845243970281  # 自动找的坐标
    # 读入FITS
    with pyfits.open(fitnm) as hdul:
        img = hdul[0].data
    # 坐标转为python索引（从0开始）
    pos = np.array([[x1, y1], [x2, y2]])
    # 孔径和背景环
    r_aper = 1.5
    r_in, r_out = 10, 15
    aper = CircularAperture(pos, r=r_aper)
    annu = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
    # 测光
    phot_table = aperture_photometry(img, [aper, annu])
    # 计算背景均值和净光度
    for i in range(2):
        mask = annu.to_mask(method='center')[i]
        annulus_data = mask.multiply(img)
        annulus_data_1d = annulus_data[(mask.data > 0)]
        bkg_mean = np.median(annulus_data_1d)
        area = aper.area
        bkg_sum = bkg_mean * area
        net_flux = phot_table['aperture_sum_0'][i] - bkg_sum
        print(f'点{i+1}: (x={pos[i,0]+1:.6f}, y={pos[i,1]+1:.6f})')
        print(f'  总光度: {phot_table["aperture_sum_0"][i]:.6f}')
        print(f'  背景均值: {bkg_mean:.6f}  背景总和: {bkg_sum:.6f}')
        print(f'  净光度: {net_flux:.6f}')
    # 绘图
    norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=SqrtStretch())
    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap='gray', origin='lower', norm=norm)
    # 分别绘制两个点的孔径和背景环，颜色区分
    aper_colors = ['lime', 'magenta']
    annu_colors = ['cyan', 'orange']
    for i in range(2):
        aper_i = CircularAperture([pos[i]], r=r_aper)
        annu_i = CircularAnnulus([pos[i]], r_in=r_in, r_out=r_out)
        aper_i.plot(color=aper_colors[i], lw=2, label=f'Aperture {i+1}')
        annu_i.plot(color=annu_colors[i], lw=1, label=f'Annulus {i+1}')
        plt.scatter(pos[i,0], pos[i,1], color=aper_colors[i], marker='+', s=80, zorder=10)
        plt.text(pos[i,0]+3, pos[i,1]+3, f'({pos[i,0]+1:.6f},{pos[i,1]+1:.6f})', color=aper_colors[i], fontsize=10)
    plt.title(os.path.basename(fitnm))
    plt.xlabel('X [pix]')
    plt.ylabel('Y [pix]')
    plt.legend()
    plt.tight_layout()
    outpng = os.path.splitext(os.path.basename(fitnm))[0] + '_twopoint_phot.png'
    plt.savefig(outpng, dpi=120)
    plt.show()


# def main2():
#     # 调用 ../code/aphot.py 的 photut_aper(img, pos, r_aper, r_in, r_out) 函数再测一次

if __name__ == '__main__':
    main()
