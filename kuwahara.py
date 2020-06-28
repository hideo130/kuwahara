import numpy as np
import cv2
from logzero import logger, loglevel
from logging import DEBUG, ERROR
from tqdm import tqdm
from pathlib import Path

do_debug = False
if do_debug:
    loglevel(DEBUG)
else:
    loglevel(ERROR)


# 公開されていたコード filtのアルゴリズムがわからないため，自分の実装もした
def kuwahara(pic, r=5, resize=False, rate=0.5):  # 元画像、正方形領域の一辺、リサイズするか、リサイズする場合の比率
    h, w, _ = pic.shape
    if resize:
        pic = cv2.resize(pic, (int(w*rate), int(h*rate)))
        h, w, _ = pic.shape
    pic = np.pad(pic, ((r, r), (r, r), (0, 0)), "edge")
    ave, var = cv2.integral2(pic)
    ave = (ave[:-r-1, :-r-1]+ave[r+1:, r+1:]-ave[r+1:, :-r-1] -
           ave[:-r-1, r+1:])/(r+1)**2  # 平均値の一括計算
    var = ((var[:-r-1, :-r-1]+var[r+1:, r+1:]-var[r+1:, :-r-1] -
            var[:-r-1, r+1:])/(r+1)**2-ave**2).sum(axis=2)  # 分散の一括計算

# --以下修正部分--
    def filt(i, j):
        return np.array([ave[i, j], ave[i+r, j], ave[i, j+r], ave[i+r, j+r]])[(np.array([var[i, j], var[i+r, j], var[i, j+r], var[i+r, j+r]]).argmin(axis=0).flatten(), j.flatten(), i.flatten())].reshape(w, h, _).transpose(1, 0, 2)
    filtered_pic = filt(
        *np.meshgrid(np.arange(h), np.arange(w))).astype(pic.dtype)  # 色の決定
    return filtered_pic


def naive_kuwahara(img, r):
    h, w, c = img.shape
    # 各チャネルの上下にr,rだけpaddingを追加．チャネル方向はpaddingしないから(0,0)
    img = np.pad(img, ((r, r), (r, r), (0, 0)), "edge")
    ave, var = cv2.integral2(img)
    ave = (ave[:-r-1, :-r-1]+ave[r+1:, r+1:]-ave[r+1:, :-r-1] -
           ave[:-r-1, r+1:])/(r+1)**2  # 平均値の一括計算
    save_ave = np.clip(ave, 0, 255).astype(np.uint8)
    cv2.imwrite("img/ave.png", save_ave)
    var = ((var[:-r-1, :-r-1]+var[r+1:, r+1:]-var[r+1:, :-r-1] -
            var[:-r-1, r+1:])/(r+1)**2-ave**2).sum(axis=2)  # 分散の一括計算
    save_var = np.clip(var, 0, 255).astype(np.uint8)
    cv2.imwrite("img/var.png", save_var)
    index_h, index_w, _ = ave.shape
    new_img = np.zeros((h, w, c))
    for i in tqdm(range(index_h - r)):
        for j in range(index_w - r):
            index = np.where(var[i:i + r + 1, j:j + r + 1] ==
                             var[i:i + r + 1, j:j + r + 1].min())
            pallete = ave[i:i+r+1, j:j+r+1]
            logger.debug(pallete)
            logger.debug(pallete.shape)
            logger.debug(index)
            chanel_b = pallete[:, :, 0]
            chanel_g = pallete[:, :, 1]
            chanel_r = pallete[:, :, 2]
            new_img[i, j] = [chanel_b[index].mean(), chanel_g[index].mean(),
                             chanel_r[index].mean()]
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    return new_img


# img = np.array(plt.imread("img/512-256.png"))
# img = cv2.imread("img/sample.jpg")
if __name__ == "__main__":
    # name = "img/512-256.png"
    # name = "img/sample2.png"
    name = "img/sample.jpg"

    img = cv2.imread(name)
    # filtered_pic = kuwahara(img, 3, False, 0.2)
    logger.debug(img.shape)
    new_img = naive_kuwahara(img, 7)
    logger.debug(new_img)
    name = Path(name).stem
    new_img = kuwahara(img, 7)
    cv2.imwrite("img/after_kuwahara_%s.png" % (name), new_img)
    cv2.imwrite("img/after_kuwahara2_%s.png" % (name), new_img)
