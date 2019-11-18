import librosa
import numpy as np
import pyworld as pw
import matplotlib.pyplot as plt
import pysptk

file_r = 'vocoder_eva/arctic_a0001_hpf.wav'
file_s = 'vocoder_eva/arctic_a0001_nwf.wav'

aud_r, sr_r = librosa.load(file_r, sr=None)
aud_s, sr_s = librosa.load(file_s, sr=None)

# print(aud_r.shape, aud_s.shape)

assert sr_r == sr_s
assert len(aud_r) == len(aud_s)

ln10_inv = 1 / np.log(10)


def eval_snr(x_r, x_s):
    # TODO: slide x_s to find max matched value
    return 10 * np.log10(np.sum(x_s ** 2) / np.sum((x_s - x_r) ** 2))


def eval_MCD(x_r, x_s):
    c_r = librosa.feature.mfcc(x_r)
    c_s = librosa.feature.mfcc(x_s)

    # plt.imshow(c_r)
    # plt.show()
    # plt.imshow(c_s)
    # plt.show()
    #
    # plt.plot(c_r[:, 20])
    # plt.plot(c_s[:, 40])
    # plt.show()
    # print((c_r- c_s))

    temp = 2 * np.sum((c_r - c_s) ** 2, axis=0)
    # print(temp)
    return 10 * ln10_inv * (temp ** 0.5)


def eval_rmse_f0(x_r, x_s, sr, frame_len='5', method='swipe'):
    if method == 'harvest':
        f0_r, t = pw.harvest(x_r.astype(np.double), sr, frame_period=100)
        f0_s, t = pw.harvest(x_s.astype(np.double), sr, frame_period=100)
    elif method == 'dio':
        f0_r, t = pw.harvest(x_r.astype(np.double), sr, frame_period=100)
        f0_s, t = pw.harvest(x_s.astype(np.double), sr, frame_period=100)
    elif method == 'swipe':
        f0_r = pysptk.sptk.swipe(x_r.astype(np.double), sr, hopsize=128)
        f0_s = pysptk.sptk.swipe(x_s.astype(np.double), sr, hopsize=128)
    elif method == 'rapt':
        f0_r = pysptk.sptk.rapt(x_r.astype(np.double), sr, hopsize=128)
        f0_s = pysptk.sptk.rapt(x_s.astype(np.double), sr, hopsize=128)
    else:
        raise ValueError('no such f0 exract method')
    # TODO: remove unvoiced frame, frame_len
    # print(f0_r, f0_s)
    f0_r_unv = (f0_r == 0) * 1
    f0_s_unv = (f0_s == 0) * 1

    voiced_mask = (1-f0_r_unv) * (1-f0_s_unv)

    print(f0_r.shape, f0_s.shape)

    y = 1200 * np.abs(np.log2(f0_r+f0_r_unv) - np.log2(f0_s+f0_s_unv))
    y = y * (1-f0_r_unv) * (1-f0_s_unv)
    # y = np.nan_to_num(y)

    f0_rmse_mean = (y * voiced_mask).sum() / voiced_mask.sum()
    precision = voiced_mask.sum() / (1 - f0_r_unv).sum()

    return f0_rmse_mean, precision


if __name__ == '__main__':
    # mcd = eval_MCD(aud_r, aud_s)
    rmse_f0 = eval_rmse_f0(aud_r, aud_s, sr_r)
    print(rmse_f0)

    # print(aud_r.shape)
    # print(eval_snr(aud_r, aud_s))
    # print(eval_snr(aud_r*10, aud_s*10))
