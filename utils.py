from typing import List, Dict
import math


def calc_conv_output_size(s: int, kernel_size: int, stride=1, padding=0, dilation=1):
    return math.floor(
        (s + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def conv_align(index: int, layers: List[Dict[str, int]]):
    ret = index
    for layer in layers:
        ret = calc_conv_output_size(ret, **layer)

    return ret if ret > 0 else 0
