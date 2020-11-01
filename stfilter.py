import numpy as np


def filter(stats, resolution):

    def edge_filter(stats):
        s_x, s_y, s_w, s_h = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
        res_x, res_y = resolution
        return stats[((res_x - s_w - s_x) * (res_y - s_h - s_y) * s_x * s_y) != 0]

    def size_filter(stats):
        return stats[np.abs(stats[:, 3] - np.max(stats[:, 3])) < frame_limit]

    def sort_L2R(stats):
        return stats[np.argsort(stats[:, 0])]

    def filter_left(stats):
        return np.min(stats[:, 0])

    def filter_top(stats):
        return np.min(stats[:, 1]) - frame_limit

    def filter_bottom(stats):
        max_h = np.max(stats[:, 3])
        max_y = np.max(stats[:, 1])
        return max_y + max_h + frame_limit

    def filter_right(stats):
        stats = stats[np.argsort(stats[:, 0])]
        char_space = stats[1:, 0] - (stats[:-1, 0] + stats[:-1, 2])
        right_index = np.argwhere(char_space > space_limit)
        if right_index.shape[0] != 0:
            stats = stats[:right_index[0][0] + 1]
        return stats

    max_char = 8
    stats = edge_filter(stats)

    if stats.shape[0] > 1:
        frame_limit = int(np.max(stats[:, 3]) / 10)
        space_limit = frame_limit * 3.8

        big = size_filter(stats)
        if big.shape[0] != 0:

            left = filter_left(big)
            top = filter_top(big)
            bottom = filter_bottom(big)

            stats = stats[stats[:, 0] >= left]
            stats = stats[stats[:, 1] >= top]
            stats = stats[(stats[:, 1] + stats[:, 3]) < bottom]
            stats = filter_right(stats)

    return stats[:max_char]
