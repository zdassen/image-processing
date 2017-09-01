# -*- coding: utf-8 -*-
#
# conv.py
#
import numpy as np


def pad(img, gap):
    """1方向にgapだけ、画像の周囲を0で埋める"""

    # 画像サイズ & パディング後の画像のサイズ
    img_rows, img_cols = img.shape
    shape_new = (img_rows + gap * 2, img_cols + gap * 2)
    img_new = np.zeros(shape_new)

    # 元の画像を埋め込む
    img_new[gap:gap+img_rows, gap:gap+img_cols] = img

    return img_new


def assign(pos_vals, img):
    """画像の指定位置に処理後の値をセットする"""
    for i, j, v in pos_vals:
        img[i, j] = v
    return img


def conv_gray(img, _filter):
    """画像の畳み込み処理を行う (グレースケールの場合)"""
    
    # フィルターの縦横サイズは同じ
    k, _ = _filter.shape
    assert k == _

    # フィルターが画像からはみ出る分
    gap = int(k / 2)

    # パディングを行う
    img_pad = pad(img, gap)

    # 画像のサイズ
    img_pad_rows, img_pad_cols = img_pad.shape

    # フィルターの位置と畳み込み後の値を保持する
    pos_vals = []

    # 処理結果となる画像
    # (パディング前のサイズ)
    img_result = np.zeros_like(img)

    # フィルターを適用する
    # (i, j) はフィルターの左上のインデックス
    # フィルターは画像をはみ出ない
    for i in range(img_pad_rows - k + 1):
        for j in range(img_pad_cols - k + 1):
            
            # フィルターの適用範囲
            target_area = img_pad[i:i+k, j:j+k]

            # 畳み込み処理を行う
            filtered_area = target_area * _filter
            sum_area = np.sum(filtered_area)

            # フィルター位置 (=パディング前の画像における位置) と
            # 畳み込み後の値を登録
            pos_vals.append((i, j, sum_area))

        # end of for j in range...
    # end of for i in range...

    # 計算後の値を代入
    img_result = assign(pos_vals, img_result)

    return img_result