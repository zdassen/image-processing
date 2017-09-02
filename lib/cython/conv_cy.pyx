# -*- coding: utf-8 -*-
#
# conv_cy.pyx
#
import numpy as np
cimport numpy as np

from applier import apply


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def sum_cy(np.ndarray[DTYPE_t, ndim=2] filtered_area):
    """フィルターの適用範囲の合計値を求める"""
    cdef unsigned int i, j
    cdef unsigned int t_rows = filtered_area.shape[0]
    cdef unsigned int t_cols = filtered_area.shape[1]
    cdef DTYPE_t sum_area = 0.0
    for i in range(t_rows):
        for j in range(t_cols):
            sum_area += filtered_area[i, j]
    return sum_area


def conv_cy(img, _filter, stride=1, padding=False, n=0):
    """画像に対して畳み込み演算を適用する"""

    img_result = apply(sum_cy, img, _filter, stride, padding, n)

    return img_result