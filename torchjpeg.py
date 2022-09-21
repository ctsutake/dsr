import math
import torch

def _normalize(N):

    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return n @ n.t()


def _harmonics(N):

    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def blockify(im, size):

    bs = im.shape[0]
    ch = im.shape[1]
    h = im.shape[2]
    w = im.shape[3]

    im = im.view(bs * ch, 1, h, w)
    im = torch.nn.functional.unfold(im, kernel_size=(size, size), stride=(size, size))
    im = im.transpose(1, 2)
    im = im.view(bs, ch, -1, size, size)

    return im


def deblockify(blocks, size):

    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]

    blocks = blocks.reshape(bs * ch, -1, int(block_size ** 2))
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks


def block_dct(blocks):

    N = blocks.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if blocks.is_cuda:
        n = n.cuda()
        h = h.cuda()

    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ blocks @ h)

    return coeff


def block_idct(coeff):

    N = coeff.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if coeff.is_cuda:
        n = n.cuda()
        h = h.cuda()

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


def batch_dct(batch):

    size = (batch.shape[2], batch.shape[3])

    im_blocks = blockify(batch, 8)
    dct_blocks = block_dct(im_blocks)
    dct = deblockify(dct_blocks, size)

    return dct


def batch_idct(coeff):

    size = (coeff.shape[2], coeff.shape[3])

    dct_blocks = blockify(coeff, 8)
    im_blocks = block_idct(dct_blocks)
    im = deblockify(im_blocks, size)

    return im


def normalize(dct, stats, channel = None):

    ch = []

    for i in range(dct.shape[1]):
        dct_blocks = blockify(dct[:, i : (i + 1), :, :], 8)

        t = ["y", "cb", "cr"][i] if channel is None else channel
        dct_blocks = stats.normalize(dct_blocks, normtype=t)

        ch.append(deblockify(dct_blocks, (dct.shape[2], dct.shape[3])))

    return torch.cat(ch, dim=1)


def denormalize(dct, stats, channel = None):

    ch = []

    for i in range(dct.shape[1]):
        dct_blocks = blockify(dct[:, i : (i + 1), :, :], 8)

        t = ["y", "cb", "cr"][i] if channel is None else channel
        dct_blocks = stats.denormalize(dct_blocks, normtype=t)

        ch.append(deblockify(dct_blocks, (dct.shape[2], dct.shape[3])))

    return torch.cat(ch, dim=1)


def quantize(dct, mat, round_func=torch.round):

    dct_blocks = blockify(dct, 8)

    if dct_blocks.is_cuda:
        mat = mat.cuda()

    quantized_blocks = round_func(dct_blocks / mat)
    quantized = deblockify(quantized_blocks, (dct.shape[2], dct.shape[3]))
    return quantized


def dequantize(dct, mat):

    dct_blocks = blockify(dct, 8)

    if dct_blocks.is_cuda:
        mat = mat.cuda()

    dequantized_blocks = dct_blocks * mat
    dequantized = deblockify(dequantized_blocks, (dct.shape[2], dct.shape[3]))
    return dequantized


luma_quant_matrix = torch.tensor([
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
])


chroma_quant_matrix = torch.tensor([
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99
])


quantization_max = 255.0


def qualities_to_scale_factors(qualities):

    qualities = qualities.clone()
    qualities[qualities <= 0] = 1
    qualities[qualities > 100] = 100

    indices_0_50 = qualities < 50
    indices_50_100 = qualities >= 50
    
    qualities[indices_0_50] = torch.div(5000, qualities[indices_0_50], rounding_mode='trunc')
    qualities = qualities.to(torch.float32)
    qualities[indices_50_100] = torch.trunc(200 - qualities[indices_50_100] * 2)
    qualities = qualities.to(torch.int64)

    return qualities


def scale_quantization_matrices(scale_factor, table="luma"):

    if table == "luma":
        t = luma_quant_matrix
    elif table == "chroma":
        t = chroma_quant_matrix

    if scale_factor.is_cuda:
        t = t.cuda()

    t = t.unsqueeze(0)
    scale_factor = scale_factor.unsqueeze(1)

    mat = torch.div(t * scale_factor + 50, 100, rounding_mode='trunc')
    mat[mat <= 0] = 1
    mat[mat > 255] = 255

    if scale_factor.is_cuda:
        mat = mat.cuda()

    return mat


def get_coefficients_for_qualities(quality, table="luma"):

    scaler = qualities_to_scale_factors(quality)
    mat = scale_quantization_matrices(scaler, table=table)
    return mat.view(-1, 1, 8, 8)


def quantize_at_quality(dct, quality, table="luma"):

    mat = get_coefficients_for_qualities(torch.tensor([quality]), table=table)
    return quantize(dct, mat)


def dequantize_at_quality(dct, quality, table="luma"):

    mat = get_coefficients_for_qualities(torch.tensor([quality]), table=table)
    return dequantize(dct, mat)


def compress_coefficients(batch, quality, table="luma"):

    batch = batch * 255 
    d = batch_dct(batch)
    d = quantize_at_quality(d, quality, table=table)
    return d


def decompress_coefficients(batch, quality, table="luma"):

    d = dequantize_at_quality(batch, quality, table=table)
    d = batch_idct(d)
    d = d / 255
    return d


