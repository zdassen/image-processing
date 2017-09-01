# -*- coding: utf-8 -*-
#
# plotter.py
#
from matplotlib import pyplot as plt


def collate(images, titles, nr, nc, show=True):
    """複数の画像をプロットする (比較用)"""

    for i, image in enumerate(images):
        plt.subplot(nr, nc, i + 1)
        
        # グレー画像の場合とそれ以外で描画方法を切り分ける
        is_gray = len(image.shape) == 2
        if is_gray:
            plt.imshow(image, cmap=plt.cm.gray)
        else:
            plt.imshow(image)

        plt.title(titles[i])

    plt.tight_layout()

    if show:
        plt.show()