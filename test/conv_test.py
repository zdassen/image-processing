# -*- coding: utf-8 -*-
#
# conv_test.py
#
import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append("../lib/")
from image_loader import load_with_orig
from filters import filter_conv_sample
from conv import conv_gray
from plotter import collate

# Cython版conv_gray()のインポート
sys.path.append("../lib/cython/")
from conv_cy import conv_gray as conv_gray_cy

# import timeit
from time import time


def test1():
    """画像の読み込み & 畳み込み演算のテスト"""
    
    # 画像を読み込む (グレースケール)
    img_type = "conv_sample"
    # img_type = "lenna"
    # img_type = "dassen_blog"
    # img_type = "cpu"
    img, original = load_with_orig(img_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 画像を確認する
    check_img = False
    if check_img:
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()

    # フィルターを生成する
    fc = filter_conv_sample()

    # 畳み込み処理を行う
    img_result = conv_gray(img, fc, stride=15, padding=False)

    # 結果を確認する
    check_result = True
    if check_result:
        images = (original, img_result)
        titles = ("Original", "Convolution")
        collate(images, titles, nr=1, nc=2)


def test2():
    """画像の読み込み & 畳み込み演算のテスト (Cython版)"""

    # 画像を読み込む (グレースケール)
    img_type = "conv_sample"
    img, original = load_with_orig(img_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # フィルターを生成する
    fc = filter_conv_sample()

    # 畳み込み処理を行う
    img_result = conv_gray_cy(img, fc)

    # 結果を確認する
    check_result = True
    if check_result:
        images = (original, img_result,)
        titles = ("Original", "Convolution",)
        collate(images, titles, nr=1, nc=2)


def test3():
    """ストライドが1より大きい場合の畳み込み演算"""

    # 画像を読み込む (グレースケール)
    img_type = "conv_sample"
    img, original = load_with_orig(img_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def perf_test():
    """畳み込み演算のパフォーマンスを比較する"""

    # 画像を読み込む (グレースケール)
    # img_type = "conv_sample"
    img_type = "lenna"
    img, original = load_with_orig(img_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # フィルターを生成する
    fc = filter_conv_sample()

    # 畳み込み処理を行う (ネイティブなPython)
    """img_copy = img.copy()
    t_native = timeit.Timer(
        "conv_gray(img_copy, fc)",
        "from conv import conv_gray"
    ).timeit()"""

    # 畳み込み処理を行う (Cython版)
    """img_copy2 = img.copy()
    t_cython = timeit.Timer(
        "conv_gray(img_copy2, fc)",
        "from conv import conv_gray as conv_gray_cy"
    ).timeit()"""

    # 結果
    """print("Native : %s(sec)" % t_native)
    print("Cython: %s(sec)" % t_cython)"""

    # ↓
    # timeで
    # ↓

    # ネイティブなPython
    img_copy = img.copy()
    start_native = time()
    img_result_native = conv_gray(img_copy, fc)
    elapsed_native = time() - start_native

    # Cython
    """img_copy_cy = img.copy()
    start_cy = time()
    img_result_cy = conv_gray_cy(img_copy_cy, fc)
    elapsed_cy = time() - start_cy"""

    # 結果
    print("Native: %s(sec)" % elapsed_native)
    # print("Cython: %s(sec)" % elapsed_cy)

    # 結果を確認する
    check_result = True
    if check_result:
        images = (original, img_result_native)
        titles = ("Original", "Native")
        collate(images, titles, nr=1, nc=2)


if __name__ == '__main__':
    test1()
    # test2()
    # test3()
    # perf_test()