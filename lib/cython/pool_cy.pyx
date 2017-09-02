# -*- coding: utf-8 -*-
#
# pool_cy.pyx
#
import numpy as np
cimport numpy as np

from applier import apply


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def pool_max(np.ndarray[DTYPE_t, ndim=2] filtered_area):
    """フィルターの適用範囲の最大値を求める (最大値プーリング)"""
    cdef unsigned int rows = filtered_area.shape[0]
    cdef unsigned int cols = filtered_area.shape[1]
    cdef unsigned int i, j
    cdef DTYPE_t max_val = 0.0
    cdef DTYPE_t tmp_val = 0.0
    for i in range(rows):
        for j in range(cols):
            tmp_val = filtered_area[i, j]
            if max_val < tmp_val:
                max_val = tmp_val
    return max_val


def pool_cy(img, k, stride=1, padding=False, n=0):
    """プーリングを行う"""

    # k x kのフィルターを生成する
    f = np.ones((k, k)).astype("float32")

    # img_result = apply(np.max, img, f, stride, padding, n)
    img_result = apply(pool_max, img, f, stride, padding, n)

    return img_result