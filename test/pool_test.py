# -*- coding: utf-8 -*-
#
# pool_test.py
#
import cv2
import numpy as np
from matplotlib import pyplot as plt
from time import time

import sys
sys.path.append("../lib/")
from image_loader import load_with_orig
from filters import filter_conv_sample
from plotter import collate

# Cython library
sys.path.append("../lib/cython/")
from conv_cy import conv_cy
from pool_cy import pool_cy


def test1():
    """プーリング処理のテスト"""

    # 画像を読み込む
    # img_type = "conv_sample"
    # img_type = "lenna"
    # img_type = "dassen_blog"
    img_type = "cpu"
    img, original = load_with_orig(img_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 画像を確認する
    check_imgs = False
    if check_imgs:
        images = (original, img)
        titles = ("Original", "Image")
        collate(images, titles, nr=1, nc=2)

    # フィルターを生成する
    f = filter_conv_sample()

    # 畳み込み演算を行う
    img_conv = conv_cy(img, f, padding=True)
    
    # 畳み込み演算の結果を確認する
    check_conved = False
    if check_conved:
        images = (original, img_conv)
        titles = ("Original", "Convolution")
        collate(images, titles, nr=1, nc=2)

    # プーリングを行う
    k = 3
    # start_pool = time()
    img_pool = pool_cy(img_conv, k, stride=3)
    # elapsed = time() - start_pool
    # print("pooling: %s (sec)" % elapsed)

    # プーリングの結果を確認する
    check_pooled = True
    if check_pooled:
        images = (original, img_conv, img_pool)
        titles = ("Original", "Convolution", "Pooling")
        collate(images, titles, nr=2, nc=2)


if __name__ == '__main__':
    test1()