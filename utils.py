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


def convert_time_to_frame_idx(time: float, conv_layers: List[Dict[str, int]], sample_rate=None, hop_length=None):
    ret = time
    if sample_rate is not None:
        ret = ret * sample_rate
    if hop_length is not None:
        ret = math.floor(ret / hop_length)
    return conv_align(ret, conv_layers)


def cut_code(
        path: str, start: float, end: float, model_config: List[Dict[str, int]], sample_rate=None,
        hop_length=None
):
    import numpy as np
    code = np.load(path)
    start_frame, end_frame = (
        convert_time_to_frame_idx(start, model_config, sample_rate, hop_length),
        convert_time_to_frame_idx(end, model_config, sample_rate, hop_length)
    )
    return code[start_frame: end_frame]
