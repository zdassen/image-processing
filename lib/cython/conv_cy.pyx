# -*- coding: utf-8 -*-
#
# conv.pyx
#
import numpy as np
cimport numpy as np


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def pad(img, gap):
    """1方向にgapだけ、画像の周囲を0で埋める"""

    # 画像サイズ & パディング後の画像のサイズ
    img_rows, img_cols = img.shape
    shape_new = (img_rows + gap * 2, img_cols + gap * 2)
    img_new = np.zeros(shape_new).astype("float32")

    # 元の画像を埋め込む
    img_new[gap:gap+img_rows, gap:gap+img_cols] = img

    return img_new


def assign(pos_vals, np.ndarray[DTYPE_t, ndim=2] img):
    """画像の指定位置に処理後の値をセットする"""
    cdef int i, j
    cdef DTYPE_t v
    for i, j, v in pos_vals:
        img[i, j] = v
    return img


def conv_gray(np.ndarray[DTYPE_t, ndim=2] img, 
    np.ndarray[DTYPE_t, ndim=2] _filter):
    """画像の畳み込み処理を行う (グレースケールの場合)"""

    # フィルターのサイズ
    cdef int k = _filter.shape[0]
    cdef int _ = _filter.shape[1]
    assert k == _

    # フィルターが画像からはみ出る分
    cdef int gap = int(k / 2)

    # パディングを行う
    img_pad = pad(img, gap)

    # 画像のサイズ (パディング後)
    cdef int img_pad_rows = img_pad.shape[0]
    cdef int img_pad_cols = img_pad.shape[1]

    # フィルターの位置と畳み込み後の値を保持する
    pos_vals = []

    # 処理結果となる画像
    # (パディング前のサイズ)
    img_result = np.zeros_like(img)

    # フィルターの適用範囲
    cdef np.ndarray[DTYPE_t, ndim=2] target_area

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