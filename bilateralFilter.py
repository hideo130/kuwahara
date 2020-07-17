import cv2
import numpy as np
from tqdm import tqdm


def naive_bilateral(img, sigma_d=1, sigma_r=1):
    h, w = img.shape
    D = int(np.ceil(3.5*sigma_d))
    img = np.pad(img, ((D, D), (D, D)), "edge")
    new_img = np.zeros((h, w))
    for i in tqdm(range(D, h+D)):
        for j in range(D, w+D):
            a = img[i, j]
            S = 0
            W = 0
            for m in range(-D, D):
                for n in range(-D, D):
                    b = img[i+m, j+n]
                    wd = np.exp(-(m**2 + n**2) / 2*sigma_d)
                    wr = np.exp(-(a**2 + b**2) / 2*sigma_r)
                    weight = wd*wr
                    S += b*weight
                    W += weight
            new_img[i-D, j-D] = S/W
    new_img = np.clip(255*new_img, 0, 255).astype(np.uint8)
    return new_img


def bilateral(img, sigma_d=1, sigma_r=1):
    D = int(np.ceil(3.5*sigma_d))
    if len(img.shape) == 2:
        h, w = img.shape
        new_img = np.zeros((h, w))
        img = np.pad(img, ((D, D), (D, D)), "edge")
    elif len(img.shape) == 3:
        h, w, _ = img.shape
        new_img = np.zeros((h, w))
        img = np.pad(img, ((D, D), (D, D), (0, 0)), "edge")

    print(h, w)
    print(w+D)
    # range filter(m**2, n**2)を予め一括計算する
    # [-D:D]の配列を作って
    tmp = np.arange(-D, D+1).reshape((1, 2*D+1))
    wr = np.exp(-(tmp**2 + tmp.T**2) / 2*sigma_r)
    print(wr)
    for i in tqdm(range(D, h+D)):
        for j in range(D, w+D):
            # domain filter(a-b)**2の方を一括計算 (2D+1，2D+1, c)の行列ができる
            wd = np.exp(-(img[i, j]-img[i-D:i+D+1, j-D:j+D+1])**2 / 2*sigma_d)
            # print(wd)
            # print(wd.shape)
            weight = wr*wd
            print(np.sum(weight, axis=(0, 1)))
            print(np.sum(w*img[i-D:i+D+1, j-D:j+D+1]))
            # カーネルの各ピクセルに重みを掛けて総和をとる．
            new_img[i-D, j-D] = np.sum(
                w*img[i-D:i+D+1, j-D:j+D+1]) / np.sum(weight, axis=(0, 1))
    new_img = np.clip(255*new_img, 0, 255).astype(np.uint8)

    return new_img


if __name__ == "__main__":
    from pathlib import Path
    name = "img/512-256.png"
    img = cv2.imread(name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)
    new_img = naive_bilateral(img_gray)
    stem = Path(name).stem
    # print(np.max(new_img))
    # cv2.imwrite("img/bilateral%s.png" % (stem), new_img)
    # cv2.imwrite("huga.png", new_img)
    # from PIL import Image
    # img = Image.fromarray(new_img)
    # img.save("hoge.png")
    bi = cv2.bilateralFilter(img_gray, 15, 20, 20)
    cv2.imwrite("img/bi2.png", bi)
