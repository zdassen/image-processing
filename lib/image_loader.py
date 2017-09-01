# -*- coding: utf-8 -*-
#
# image_loader.py
#
import cv2
import numpy as np


def create_dot():
    """サンプル用のドット (小) 画像を生成する"""
    img = np.zeros((7, 7)).astype("float32")
    img[3, 3] = .9
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_dot_4x3():
    """サンプル用のドット (小) 画像を生成する"""
    img = np.zeros((4, 3)).astype("float32")
    img[2, 1] = .9
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_horizontal_line_4x3():
    """4 x 3の横線の画像を生成する"""
    img = np.array([
        [.0, .0, .0],
        [.0, .0, .0],
        [.9, .9, .9],
        [.0, .0, .0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def create_dot_bottom_3x3():
    """サンプル用のドット (小) 画像を末尾行に生成する"""
    img = np.zeros((3, 3)).astype("float32")
    img[2, 1] = .9
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_dot_5x5():
    """ドット画像を生成する (画像サイズが4 x 4)"""
    img = np.array([
        [.0, .0, .0, .0, .0],
        [.0, .9, .9, .9, .0],
        [.0, .9, .9, .9, .0],
        [.0, .9, .9, .9, .0],
        [.0, .0, .0, .0, .0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_big_dot():
    """サンプル用のドット画像を生成する"""
    img = np.array([
        [.0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .9, .9, .9, .0, .0],
        [.0, .0, .9, .9, .9, .0, .0],
        [.0, .0, .9, .9, .9, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_big_dot_noised():
    """サンプル用のドット (大) 画像を生成する (ノイズあり)"""
    img = create_big_dot()
    img[3, 6] = 0.9
    img[6, 4] = 0.7
    return img


def create_horizontal_line():
    """横線の画像を生成する"""
    img = np.array([
        [.0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0],
        [.9, .8, .9, .8, .9, .8, .9],
        [.0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_horizontal_line_noised():
    """横線の画像を生成する (ノイズあり)"""
    img = create_horizontal_line()
    img[6, 4] = .8
    return img


def create_vertical_line():
    """縦線の画像を生成する"""
    img = np.array([
        [.0, .0, .0, .9, .0, .0, .0],
        [.0, .0, .0, .8, .0, .0, .0],
        [.0, .0, .0, .7, .0, .0, .0],
        [.0, .0, .0, .8, .0, .0, .0],
        [.0, .0, .0, .9, .0, .0, .0],
        [.0, .0, .0, .8, .0, .0, .0],
        [.0, .0, .0, .7, .0, .0, .0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_vertical_line_noised():
    """縦線の画像を生成する (ノイズあり)"""
    img = create_vertical_line()
    img[3, 6] = 0.9
    img[6, 4] = 0.7
    return img


def create_dot_grad():
    """勾配を持ったドット画像を生成する"""
    img = np.array([
        [.0, .0, .0, .0],
        [.0, .9, .6, .0],
        [.0, .6, .9, .0],
        [.0, .0, .0, .0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_mount_grad():
    """グラデーション画像を生成する"""
    img = np.array([
        [.4, .5, .6, .7, .6, .5, .4],
        [.4, .5, .6, .7, .6, .5, .4],
        [.4, .5, .6, .7, .6, .5, .4],
        [.3, .4, .5, .6, .5, .4, .3],
        [.1, .3, .5, .7, .5, .3, .1],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_pyramid():
    """ピラミッド状 (上から見た場合) の画像を生成する"""
    img = np.array([
        [.2, .2, .2, .6, .2, .2, .2],
        [.2, .2, .6, .7, .6, .2, .2],
        [.2, .6, .7, .8, .7, .6, .2],
        [.6, .7, .8, .9, .8, .7, .6],
        [.2, .6, .7, .8, .7, .6, .2],
        [.2, .2, .6, .7, .6, .2, .2],
        [.2, .2, .2, .6, .2, .2, .2],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_edge_0():
    """x軸に対して0°のエッジを持つ画像を生成する"""
    img = np.array([
        [.0, .0, .0, .0],
        [.9, .9, .0, .0],
        [.0, .0, .9, .9],
        [.0, .0, .0, .0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_edge_45():
    """x軸に対して45°のエッジを持つ画像を生成する"""
    img = np.array([
        [1.0,   .0,    .0,    .0],
        [  .0, 1.0,    .0,    .0],
        [  .0,    .0, 1.0,    .0],
        [  .0,    .0,    .0, 1.0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_mount_ridge(is_sharp=False):
    """山の尾根 (のような) 画像を生成する"""
    img = np.array([
        [.4, .5, .6, .7, .6, .5, .4],
        [.4, .5, .6, .7, .6, .5, .4],
        [.4, .5, .6, .7, .6, .5, .4],
        [.4, .5, .6, .7, .6, .5, .4],
        [.4, .5, .6, .7, .6, .5, .4],
        [.4, .5, .6, .7, .6, .5, .4],
        [.4, .5, .6, .7, .6, .5, .4],
    ]).astype("float32")

    # 尾根を尖らせる
    if is_sharp:
        img[:, 3] = .8

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_mount_ridge_sharp():
    """尖った山の尾根 (のような) 画像を生成する"""
    return create_mount_ridge(is_sharp=True)


def create_square():
    """四角形の画像を生成する"""
    img = np.array([
        [.1, .0, .1, .0, .1, .0, .1],
        [.0, .9, .8, .9, .8, .9, .0],
        [.1, .8, .1, .0, .1, .8, .1],
        [.0, .9, .0, .1, .0, .9, .0],
        [.1, .8, .1, .0, .1, .8, .1],
        [.0, .9, .8, .9, .8, .9, .0],
        [.1, .0, .1, .0, .1, .0, .1],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def create_conv_sample():
    """畳み込み処理用のサンプル"""
    img = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
    ]).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_lenna_gray():
    """グレーのLenna画像を読み込む"""
    img_type = "lenna_original"
    img = load(img_type)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def load_image(fpath):
    """指定パスの画像を読み込む"""

    # パスが存在しなくとも例外は発生しない
    img = cv2.imread(fpath)
    
    # 代わりに、画像がNoneなら例外とする
    try:
        img.shape
    except Exception as e:
        print(e)
    else:
        return img


def load(typestr):
    """指定された画像を読み込む"""

    # 指定可能な画像の種類とパス
    prefix = "../images/%s"
    paths = {
        "cpu": prefix % "cpu.png",
        "lenna": prefix % "lenna.jpg",
        "lenna_original": prefix % "lenna_original.png",
        "dassen_blog": prefix % "dassen_blog.jpg",
    }
    availables = paths.keys()

    # 一般の画像を読み込む
    if typestr in availables:
        path = paths[typestr]
        return load_image(path)

    # NumPy.ndarrayで自作した画像を読み込む
    else:
        original_funcs = {
            "dot": create_dot,
            "dot_4x3": create_dot_4x3,
            "horizontal_line_4x3": create_horizontal_line_4x3,
            "dot_bottom_3x3": create_dot_bottom_3x3,
            "dot_5x5": create_dot_5x5,
            "big_dot": create_big_dot,
            "big_dot_noised": create_big_dot_noised,
            "horizontal_line": create_horizontal_line,
            "horizontal_line_noised": create_horizontal_line_noised,
            "vertical_line": create_vertical_line,
            "vertical_line_noised": create_vertical_line_noised,
            "lenna_gray": load_lenna_gray,
            "dot_grad": create_dot_grad,
            "mount_grad": create_mount_grad,
            "pyramid": create_pyramid,
            "edge_0": create_edge_0,
            "edge_45": create_edge_45,
            "mount_ridge": create_mount_ridge,
            "mount_ridge_sharp": create_mount_ridge_sharp,
            "square": create_square,
            "conv_sample": create_conv_sample,
        }
        availables_original = original_funcs.keys()
        if typestr in availables_original:
            creater = original_funcs[typestr]
            return creater()
        else:
            emsg = "couldn't find image creater function"
            raise ValueError(emsg)


def load_with_orig(typestr):
    """指定された画像を読み込む & 退避用の画像も取得する"""
    img = load(typestr)
    original = img.copy()
    return img, original