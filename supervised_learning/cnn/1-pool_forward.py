#!/usr/bin/env python3
"""Same as before but now with pooling!"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode="max"):
    """A_prev is an array of shape (
    m, h_prev, w_prev, c_prev) containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw)
      containing the size of the kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg,
      indicating whether to perform maximum or average pooling, respectively
    Returns: the output of the pooling layer"""

    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride

    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_prev):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    A_slice = A_prev[i, h_start:h_end, w_start:w_end, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(A_slice)
                    elif mode == "avg":
                        A[i, h, w, c] = np.mean(A_slice)

    return A
