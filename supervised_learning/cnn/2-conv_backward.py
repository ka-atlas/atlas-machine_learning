import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    (m, h_new, w_new, c_new) = dZ.shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    (sh, sw) = stride
    
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    
    if padding == "same":
        ph = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pw = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0
    
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode="constant")
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode="constant")
    
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw
                    A_slice = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                    
                    dA_prev_pad[i, h_start:h_end, w_start:w_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += A_slice * dZ[i, h, w, c]
    
    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad
    
    return dA_prev, dW, db
