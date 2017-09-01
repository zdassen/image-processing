# -*- coding: utf-8 -*-
#
# filters.py
#
import numpy as np


def diff_h_3x3(rev=False):
    """3 x 3の微分フィルターを生成する (横方向に微分)"""
    df = np.array([
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0],
    ]).astype("float32")
    
    # 微分する方向を逆にする (右から左)
    if rev: df = np.fliplr(df)

    return df


def diff_v_3x3(rev=False):
    """3 x 3の微分フィルターを生成する (縦方向に微分)"""
    df = np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0]
    ]).astype("float32")

    # 微分する方向を逆にする (下から上)
    if rev: df = np.flipud(df)

    return df


def prewitt_h_3x3(rev=False):
    """3 x 3のPrewittフィルターを生成する (横方向に微分)"""
    pf = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]).astype("float32")

    # 指定がある場合は逆方向に微分する (右から左)
    if rev: pf = np.fliplr(pf)

    return pf


def prewitt_v_3x3(rev=False):
    """3 x 3のPrewittフィルターを生成する (縦方向に微分)"""
    pf = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ]).astype("float32")

    # 指定がある場合は逆方向に微分する (上から下)
    if rev: pf = np.flipud(pf)

    return pf


def sobel_h_3x3(rev=False):
    """3 x 3のSobelフィルターを生成する (横方向に微分)"""
    sf = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype("float32")

    # 指定がある場合は逆方向に微分する (右から左)
    if rev: sf = np.fliplr(sf)

    return sf


def sobel_v_3x3(rev=False):
    """3 x 3のSobelフィルターを生成する (縦方向に微分)"""
    sf = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]).astype("float32")

    # 指定がある場合は逆方向に微分する (下から上)
    if rev: sf = np.flipud(sf)

    return sf


def scharr_h_3x3(rev=False):
    """3 x 3のScharrフィルターを生成する (横方向に微分)"""
    sf = np.array([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ]).astype("float32")

    # 指定がある場合は逆方向に微分する (右から左)
    if rev: sf = np.fliplr(sf)

    return sf


def scharr_v_3x3(rev=False):
    """3 x 3のScharrフィルターを生成する (縦方向に微分)"""
    sf = np.array([
        [-3, -10, -3],
        [0, 0, 0],
        [3, 10, 3],
    ]).astype("float32")

    # 指定がある場合は逆方向に微分する (右から左)
    if rev: sf = np.flipud(sf)

    return sf


def filter_conv_sample():
    """畳み込み処理用のフィルター (サンプル)"""
    f = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ]).astype("float32")
    return f