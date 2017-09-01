# -*- coding: utf-8 -*-
#
# conv_test.py
#
import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append("../lib")
from image_loader import load_with_orig
from filters import filter_conv_sample
from conv import conv_gray


def test1():
    """画像の読み込み & 畳み込み演算のテスト"""
    
    # 画像を読み込む (グレースケール)
    img_type = "conv_sample"
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
    img_result = conv_gray(img, fc)

    # 結果を確認する
    check_result = False
    if check_result:
        plt.imshow(img_result, cmap=plt.cm.gray)
        plt.show()


if __name__ == '__main__':
    test1()