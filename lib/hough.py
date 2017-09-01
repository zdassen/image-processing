# -*- coding: utf-8 -*-
#
# hough.py
#
import numpy as np
from matplotlib import pyplot as plt


def perpendicular_length(xi, yi, theta_a):
    """
    xy座標系における点 (x, y) を通る直線に、原点から垂線を
    下ろした場合の垂線の長さを求める
    """

    # matplotlibにおいてはインデックス = 座標でよい
    x, y = xi, yi

    # x軸と (x軸、原点→(x, y))がなす角を求める
    tan_theta = np.arctan2(x, y)
    theta = tan_theta / np.pi * 180

    # 極座標におけるr (半径)
    r = np.sqrt(x ** 2 + y ** 2)

    # θ >= αならθ - α、θ < αならα - θ
    if theta >= theta_a:
        diff = theta - theta_a
    else:
        diff = theta_a - theta

    # 垂線の長さを調べる
    # p = r * cos(θ - α) より
    rad_theta = diff / 180 * np.pi    # ラジアンに直す
    p = r * np.cos(rad_theta)

    return p


def plot_p_theta(points_i):
    """p - θ空間をプロットする"""
    
    # 1°間隔で変化を観察する
    PI = 90
    theta_a = range(0, PI + 1, 1)

    # 画像上のそれぞれの点について、
    # p - θの関係をプロットする
    for pi in points_i:
        xi, yi = pi
        ps = [
            perpendicular_length(xi, yi, tha) for tha in theta_a
        ]
        plt.plot(theta_a, ps)

    # 軸のラベルを指定する
    plt.xlabel(r"$\theta$")
    plt.ylabel("p")

    # 値の範囲を設定する
    plt.xlim((0, PI))

    plt.tight_layout()
    plt.grid(True, alpha=0.4)
    plt.show()