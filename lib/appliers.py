# -*- coding: utf-8 -*-
#
# appliers.py
#
import numpy as np


def assign(img, interests):
    """注目画素に値を設定する"""
    for i, j, c, v in interests:
        img[i, j, c] = v
    return img


def apply_filter(img, _filter):
    """画像にフィルターを適用する (パディングなし)"""

    # 画像 & フィルターのサイズ
    img_rows, img_cols, rgb = img.shape
    k, _ = _filter.shape
    assert k == _

    # フィルターの端から注目画素までの画素の数
    gap = int(k / 2.0)

    # 処理結果となる画像
    img_result = np.zeros_like(img)

    # 注目画素のインデックス & セットされるべき値
    interests = []

    # フィルターを適用する
    # 以下は注目画素のインデックスが (i, j) の場合
    for c in range(rgb):
        for i in range(gap, gap + img_rows - gap * 2):
            for j in range(gap, gap + img_cols - gap * 2):

                # フィルターの適用範囲 (画像における範囲)
                i_from = i - gap
                i_to = i + gap
                j_from = j - gap
                j_to = j + gap

                # フィルターの適用範囲
                target_area = img[i_from:i_to+1, j_from:j_to+1, c]

                # フィルターの適用範囲に対して処理を行う
                interest_val = np.sum(target_area * _filter)
                interests.append((i, j, c, interest_val))

            # end of for j in range...
        # end of for i in range...
    # end of for c in range...

    # 画素値を設定する
    img_result = assign(img, interests)

    return img_result


def apply_filters(img, filter_horizontal, filter_vertical):
    """画像からフィルターがはみ出ない範囲でフィルターを適用する"""

    assert filter_horizontal.shape == filter_vertical.shape

    # 画像 & フィルターのサイズ
    img_rows, img_cols, rgb = img.shape
    k, _ = filter_horizontal.shape
    assert k == _

    # フィルターの端から注目画素までの画素の数
    gap = int(k / 2.0)

    # 処理結果となる画像
    img_result = np.zeros_like(img)

     # 注目画素のインデックス & セットされるべき値
    interests = []

    # フィルターを適用する (注目画素のインデックスが (i, j) の場合)
    for c in range(rgb):
        for i in range(gap, gap + img_rows - gap * 2):
            for j in range(gap, gap + img_cols - gap * 2):

                # フィルターの適用範囲 (画像における範囲)
                i_from = i - gap
                i_to = i + gap
                j_from = j - gap
                j_to = j + gap

                # フィルターの適用範囲
                target_area = img[i_from:i_to+1, j_from:j_to+1, c]
                assert target_area.shape == filter_horizontal.shape

                # フィルターの適用範囲に対して処理を行う (横)
                interest_val_h = np.sum(target_area * filter_horizontal)
                
                # フィルターの適用範囲に対して処理を行う (縦)
                interest_val_v = np.sum(target_area * filter_vertical)

                # 注目画素の値を求める
                interest_val = np.sqrt(interest_val_h ** 2 + interest_val_v ** 2)
                interests.append((i, j, c, interest_val))

            # end of for j in range...
        # end of for i in range...
    # end of for c in range...

    # 画素値を設定する
    img_result = assign(img, interests)

    return img_result


def apply_filter_f(img, _filter, func):
    """画像にフィルターを適用する (パディングなし)"""

    # 画像 & フィルターのサイズ
    img_rows, img_cols, rgb = img.shape
    k, _ = _filter.shape
    assert k == _

    # フィルターの端から注目画素までの画素の数
    gap = int(k / 2.0)

    # 処理結果となる画像
    img_result = np.zeros_like(img)

    # 注目画素のインデックス & セットされるべき値
    interests = None

    # フィルターを適用する
    # 以下は注目画素のインデックスが (i, j) の場合
    for c in range(rgb):
        for i in range(gap, gap + img_rows - gap * 2):
            for j in range(gap, gap + img_cols - gap * 2):

                # フィルターの適用範囲 (画像における範囲)
                i_from = i - gap
                i_to = i + gap
                j_from = j - gap
                j_to = j + gap

                # フィルターの適用範囲
                target_area = img[i_from:i_to+1, j_from:j_to+1, c]

                #
                # ※抽象化は難しいかな? (170825 10:06:00)
                #
                # フィルターの適用範囲に対して処理を行う
                # interests = func(target_area, )

            # end of for j in range...
        # end of for i in range...
    # end of for c in range...

    # 画素値を設定する
    """img_result = assign(img, interests)

    return img_result"""