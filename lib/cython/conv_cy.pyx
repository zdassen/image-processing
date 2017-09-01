# -*- coding: utf-8 -*-
#
# conv_cy.pyx
#
import numpy as np
cimport numpy as np


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def pad(img, gap, n=0):
    """各方向にgapだけ、画像の周囲をnで埋める"""

    # 画像サイズ & パディング後の画像のサイズ
    img_rows, img_cols = img.shape
    shape_new = [
        size + gap * 2 for size in (img_rows, img_cols)
    ]

    # 新しい画像
    img_new = np.ones(shape_new).astype("float32") * n

    # 元の画像を埋め込む
    img_new[gap:gap+img_rows, gap:gap+img_cols] = img

    return img_new


def assign(pos_vals, img):
    """画像の指定位置に処理後の値をセットする"""
    for i, j, v in pos_vals:
        img[i, j] = v
    return img


cdef sum_cy(np.ndarray[DTYPE_t, ndim=2] filtered_area):
    cdef unsigned int i, j
    cdef unsigned int t_rows = filtered_area.shape[0]
    cdef unsigned int t_cols = filtered_area.shape[1]
    cdef DTYPE_t sum_area = 0.0
    for i in range(t_rows):
        for j in range(t_cols):
            sum_area += filtered_area[i, j]
    return sum_area


def conv_gray_cy(img, _filter, stride=1, padding=False, n=0):
    # 画像の畳み込み処理を行う (グレースケールの場合)

    # ※ここだけやってみる
    cdef np.ndarray[DTYPE_t, ndim=2] target_area
    cdef DTYPE_t sum_area

    # フィルターの縦横サイズは同じ
    k, _ = _filter.shape
    assert k == _

    # 元の画像サイズ
    img_rows, img_cols = img.shape

    # フィルターが画像からはみ出る分
    gap = int(k / 2)

    # パディングを行う場合

    if padding:
        img_pad = pad(img, gap, n=n)
    else:
        img_pad = img.copy()

    # パディング後の画像サイズ (行った場合のみ)
    img_pad_rows, img_pad_cols = img_pad.shape

    # 畳み込み後の位置と値を保持する
    pos_vals = []

    # 畳み込み後の、値がセットされる位置 (行)
    # 畳み込み演算によって求められた値の数だけ
    # 行数、列数があればよいのでカウントする
    # strideで飛び飛びに計算してもカウントすれば足りる
    i_cur = 0

    # フィルターを適用する
    # (i, j) はフィルターの左上のインデックス
    for i in range(0, img_pad_rows - k + 1, stride):

        # 畳み込み後の、値がセットされる位置 (列)
        j_cur = 0

        for j in range(0, img_pad_cols - k + 1, stride):

            # フィルターの適用範囲
            target_area = img_pad[i:i+k, j:j+k]

            # 畳み込み処理を行う
            filtered_area = target_area * _filter
            # sum_area = np.sum(filtered_area)
            # ↓
            sum_area = sum_cy(filtered_area)

            # 畳み込み後の位置、値を記録させる
            pos_vals.append((i_cur, j_cur, sum_area))
            j_cur += 1

        # end of for j in range...

        i_cur += 1

    # end of for i in range...

    # 処理結果の画像のサイズ
    result_shape = (i_cur, j_cur)
    assert len(pos_vals) == i_cur * j_cur

    # 処理結果の画像
    img_result = np.zeros(result_shape).astype("float32")

    # 計算後の値を代入
    img_result = assign(pos_vals, img_result)

    return img_result