import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    assert kc == c, "Kernel channels must match image channels"

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("Invalid padding")

    # Pad the images
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Compute output shape
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    # Convolution with 2 loops
    for i in range(out_h):
        for j in range(out_w):
            region = images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
