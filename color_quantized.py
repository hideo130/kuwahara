from PIL import Image
import io
import PySimpleGUI as sg
import numpy as np
from logzero import logger
import copy

def call_img(img):
    """
    input img:Image object
    return img:画像のbytes型
    sg.ImageはPNGかGIFしか読み込むことができない．
    レンダリングしたpng画像やハイパースペクトル画像のpng形式のbytes型を作成し，
    その後，sg.Imageの引数dataに渡すことで画像を表示している．
    """

    bio = io.BytesIO()
    # バッファに画像を出力
    img.save(bio, format="PNG")
    # cv2.imwrite("tmp.png", img)
    # バッファの全内容の bytes型 をgetvalue()で取得する
    img = bio.getvalue()
    return img


def quantize(img, bin):
    deltaq = 255/bin

    new_img = copy.deepcopy(img)
    for i in range(bin):
        q = (i+1)*deltaq
        # new_img = np.where(i*deltaq < img and img < q, q + (deltaq / 2)
        #                    * np.tanh(img - q), img)
        index = i*deltaq < img & img < q
        new_img[index] = q + (deltaq / 2) * np.tanh(img[index] - q)

    return new_img


if __name__ == "__main__":

    name = "img/sample.jpg"
    img = Image.open(name)
    show_img = call_img(img)
    img = np.array(img)
    quantized_img = quantize(img, 8)
    quantized_img = Image.fromarray(quantized_img)
    quantized_img = call_img(quantized_img)
    layout = [
        [sg.Slider(range=(0, 254), orientation='h', enable_events=True,
                   size=(34, 20), key='__SLIDER1__', default_value=250)],
        # [sg.Slider(range=(1, 255), orientation='h', enable_events=True,
        #            size=(34, 20), key='__SLIDER2__', default_value=200)],
        [sg.Image(data=show_img)], [sg.Image(data=quantized_img,  key='_OUTPUT_')]]
    window = sg.Window('白飛びチェック', layout, finalize=True)

    while True:
        event, values = window.read()
        logger.info(event)
        if event in (None, 'Quit'):
            print('exit')
            break
    