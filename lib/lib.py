# -*- coding: utf-8 -*-
#
# lib.py
#
import cv2
import numpy as np


def load_bw_image():
    """白黒の画像データを読み込む"""

    img = np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ]).astype("float32")

    return img


def load_title():
    """ブログのタイトル画像を読み込む"""

    img_path = "../images/dassen_blog.jpg"
    img = cv2.imread(img_path, 0).astype("float32")

    return img


def smooth(img, i, j, kernel):
    """
    カーネルの左上のインデックスが (i, j) となるよう
    カーネルを適用する
    """ 
    
    # カーネルのサイズ
    rows, cols = kernel.shape

    # カーネルで平滑化する
    kerneled = img[i:i+rows, j:j+cols] * kernel
        
    # 注目画素の位置 (i, jからのギャップ)
    interest_gap = int(rows / 2)    # 切り捨て

    # 注目画素の色を変える
    img[i+interest_gap, j+interest_gap] = np.sum(kerneled)

    return img


def walk(img, kernel):
    """画像全体にカーネル (フィルター) を適用する"""

    img_rows, img_cols = img.shape
    k, _ = kernel.shape

    # 画像よりも大きなフィルタは適用できない
    # (フィルタの縦、横いずれかが画像より大きい場合はエラー)
    if img_rows < k or img_cols < k:
        emsg = "kernel size must be less than the size of the image"
        raise ValueError(emsg)

    # 画像全体にカーネルを適用する
    # カーネル (フィルター) は縦横にそれぞれ、縦(横) - k + 1だけ動ける
    # (i, j) はカーネル (フィルター) の左上のインデックスとなる
    for i in range(img_rows - (k - 1)):
        for j in range(img_cols - (k - 1)):
            img = smooth(img, i, j, kernel) 

    return img


def dist_from_center(i, j, k):
    """カーネルの中心位置 (注目画素) と 画素 (i, j) との距離を得る"""

    # 注目画素の位置 (インデックス)
    center_i = int(k / 2)
    center_ij = np.array([center_i, center_i])

    # 周辺画素の位置
    p = np.array([i, j])

    return np.sqrt(np.sum((p - center_ij) ** 2))


def _g(dist, k, sigma=None):
    """
    ガウス分布を用いて、注目画素との距離から
    画素の重みを得る
    """

    # sigma (画素値の標準偏差) を目安に基づいて設定する
    # 要は、フィルタサイズが大きくなればsigmaも大きくなる
    # (分布が滑らかになる = ぼかしが強力になる)
    if not sigma:
        sigma = int((k - 1) / 2.0)
    else:
        sigma = 1.3

    a = 1.0 / 2.0 / np.pi / (sigma ** 2)
    p = -1.0 * (dist ** 2) / 2.0 / (sigma ** 2)
    b = np.exp(p)
    return a * b


def create_g_kernel(k, sigma=None):
    """k x kのガウスカーネルを生成する"""

    g_kernel = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            d = dist_from_center(i, j, k)
            g_kernel[i, j] = g(d, k, sigma)

    # 正規化する
    # 正規化を行わないと画像が暗くなっていくので
    # 元の画像と明るさを一致させるために
    # カーネルの重みの合計を1.0に揃える
    z = np.sum(g_kernel)
    g_kernel /= z
    assert np.sum(g_kernel) == 1.0

    return g_kernel


def check_sigma(sigma, k):
    """
    標準偏差が指定されているかどうかチェックする
    指定されていない場合は、最適な標準偏差の値を求める
    """
    if not sigma:
        sigma = int((k - 1) / 2.0)
    else:
        sigma = sigma

    return sigma


def create_bi_kernel(img, i, j, k, sigma1=None, sigma2=None):
    """
    左上がインデックス (i, j) になるようなカーネルを生成する
    (バイラテラルフィルター)
    """

    # 標準偏差を設定する (指定がない場合)
    sigma1 = check_sigma(sigma1, k)
    sigma2 = check_sigma(sigma2, k)

    # 注目画素を取得
    # (interest_gapは (i, j) からのインデックス差)
    interest_gap = int(k / 2)
    interest = img[i+interest_gap, j+interest_gap]

    # 注目画素の位置 (インデックス)
    interest_ij = np.array([i+interest_gap, j+interest_gap])

    # カーネルを生成
    kernel = np.zeros((k, k))
    for m in range(k):
        for n in range(k):

            # 周辺画素を取得
            target = img[i + m, j + n]

            # 周辺画素の位置 (インデックス)
            target_ij = np.array([i + m, j + n])

            # 注目画素との距離を算出
            d = np.sqrt(np.sum((interest_ij - target_ij) ** 2))

            # 注目画素と周辺画素との距離から重みを算出
            a = g(d, k, sigma1)

            # 注目画素と周辺画素の輝度差を算出
            ldist = np.sqrt((interest - target) ** 2)

            # 輝度差から重みを算出
            b = g(ldist, k, sigma2)

            # カーネルに値を設定
            kernel[m, n] = a * b

        # end of for n in range...
    # end of for m in range...

    # 重みの合計を1に揃える (正規化)
    # 丸め誤差の関係でround()している
    z = np.sum(kernel)
    kernel /= z
    assert np.round(np.sum(kernel), 5) == 1.0

    return kernel


def g(dist, k, sigma):
    """
    注目画素と周辺画素との距離に応じた重みを算出する
    or
    注目画素と周辺画素との輝度差に応じた重みを算出する
    """

    p = -1.0 * (dist ** 2) / 2.0 / (sigma ** 2)
    return np.exp(p)


"""
def lum_dist(img, i, j, k):
    # 周辺画素 (i, j) と注目画素との輝度差を求める

    # 注目画素の輝度を取得する
    interest_gap = int(k / 2)
    interest = img[i+interest_gap, j+interest_gap]

    # 周辺画素の輝度
    img_ij = img[i, j]

    return np.sqrt(np.sum((interest - img_ij) ** 2))
"""


"""
def g_lum(ldist, k, sigma):
    # 注目画素と周辺画素との輝度差に応じた重みを算出する

    p = -1.0 * (ldist ** 2) / 2.0 / (sigma ** 2)
    return np.exp(p)
"""


def interests_on(img, interests):
    """対象となる全ての注目画素の値を1にする"""

    # 注目画素の値を1にする
    for i, j in interests:
        img[i, j] = 1

    return img


def interests_off(img, interests):
    """対象となる全ての注目画素の値を0にする"""

    for i, j in interests:
        img[i, j] = 0

    return img


def pad(img, gap):
    """
    画像からカーネルがはみ出る分 (gap) だけ周囲をゼロで埋める
    画像のサイズは (gap + width + gap, gap + height + gap) となる
    """

    # 元の画像のサイズ
    rows, cols = img.shape

    # ゼロで埋めた後のサイズ
    rows_after, cols_after = [n + (gap * 2) for n in (rows, cols)]

    # 縦、横にgapだけズラして画像を埋め込む (OR演算)
    img_after = np.zeros((rows_after, cols_after)).astype("uint8")
    # img_after[gap:gap+rows, gap:gap+cols] |= img
    img_after[gap:gap+rows, gap:gap+cols] = img

    return img_after


def cut(img, gap):
    """画像の上下左右 (pad()した分) をgapだけくり抜く"""

    # くり抜いた後の画像のサイズ
    rows, cols = [n - gap * 2 for n in img.shape]

    return img[gap:gap+rows, gap:gap+cols]


def process(img, kernel, mode):
    """膨張or収縮処理を行う"""

    # 処理モード (膨張 or 収縮) をチェック
    if mode not in ("dilate", "erode"):
        emsg = "process mode must be \"dilate\" or \"erode\""
        raise ValueError(emsg)

    # 画像のサイズ
    rows, cols = img.shape

    # カーネルのサイズ (縦 = 横とする)
    k, _ = kernel.shape
    assert k == _

    # img[i, j]からの注目画素の位置のズレ
    gap = int(k / 2)

    # 画像+gap x 2よりカーネルが大きい場合はエラー
    if rows + gap * 2 < k or cols + gap * 2 < k:
        emsg = "kernel size must be less than the size of the image"
        raise ValueError(emsg)

    # 画像の周囲をゼロで埋める
    img_after = pad(img, gap)
    rows_after, cols_after = img_after.shape

    # 値を設定する注目画素の位置を記憶させる
    interests = []

    # 画像にカーネルを適用する
    # 移動範囲は rows_after (cols_after) - k + 1まで
    for i in range(rows_after - (k - 1)):
        for j in range(cols_after - (k - 1)):

            # AND演算を行う
            kerneled = img_after[i:i+k, j:j+k] & kernel

            if mode == "dilate":
                
                # 膨張処理を行う
                # 一つでも1があった場合、注目画素の位置 (インデックス)
                # を記憶させておく
                # 画像をその都度書き換えると、次の段階でのカーネル適用で
                # 書き換えたばかりのビットに影響を受けるため
                # 書き換えるべき注目画素のインデックスだけを記憶するようにする
                if np.sum(kerneled) > 0:
                    interests.append((i+gap, j+gap))

            elif mode == "erode":

                # 収縮処理を行う
                # 一つでも0があった場合、注目画素の位置 (インデックス)
                # を記憶させておく
                if np.sum(kerneled) < k * k:
                    interests.append((i+gap, j+gap))

        # end of for j in range...
    # end of for i in range...

    if mode == "dilate":

        # 注目画素に1を設定する (膨張処理)
        img_after = interests_on(img_after, interests)

    elif mode == "erode":

        # 注目画素に0を設定する (収縮処理)
        img_after = interests_off(img_after, interests)

    # ゼロで埋めた分をくり抜く
    img_after = cut(img_after, gap)

    return img_after


def dilate(img, kernel):
    """膨張処理を行う"""

    return process(img, kernel, mode="dilate")


def erode(img, kernel):
    """収縮処理を行う"""

    return process(img, kernel, mode="erode")