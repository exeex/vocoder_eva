import librosa
import numpy as np
import pyworld as pw
# import matplotlib.pyplot as plt
import pysptk

ln10_inv = 1 / np.log(10)


def pad_to(x, target_len):
    pad_len = target_len - len(x)

    if pad_len <= 0:
        return x[:target_len]
    else:
        return np.pad(x, (0, pad_len), 'constant', constant_values=(0, 0))


def eval_snr(x_r, x_s):
    # TODO: slide x_s to find max matched value 原論文有做滑動x_s，找到最大匹配的snr值，這邊還沒實作
    return 10 * np.log10(np.sum(x_s ** 2) / np.sum((x_s - x_r) ** 2))


def eval_MCD(x_r, x_s):
    # TODO: verify value 確認做出來的值是否正確 (和原論文比較)
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


def eval_rmse_f0(x_r, x_s, sr, frame_len='5', method='swipe', tone_shift=None):
    # TODO: 要可以改動 frame len (ms) 或者 hop_size
    if method == 'harvest':
        f0_r, t = pw.harvest(x_r.astype(np.double), sr, frame_period=50)
        f0_s, t = pw.harvest(x_s.astype(np.double), sr, frame_period=50)
    elif method == 'dio':
        f0_r, t = pw.dio(x_r.astype(np.double), sr, frame_period=50)
        f0_s, t = pw.dio(x_s.astype(np.double), sr, frame_period=50)
    elif method == 'swipe':
        f0_r = pysptk.sptk.swipe(x_r.astype(np.double), sr, hopsize=128)
        f0_s = pysptk.sptk.swipe(x_s.astype(np.double), sr, hopsize=128)
    elif method == 'rapt':
        f0_r = pysptk.sptk.rapt(x_r.astype(np.double), sr, hopsize=128)
        f0_s = pysptk.sptk.rapt(x_s.astype(np.double), sr, hopsize=128)
    else:
        raise ValueError('no such f0 exract method')

    # length align
    f0_s = pad_to(f0_s, len(f0_r))

    # make unvoice / vooiced frame mask
    f0_r_uv = (f0_r == 0) * 1
    f0_r_v = 1 - f0_r_uv
    f0_s_uv = (f0_s == 0) * 1
    f0_s_v = 1 - f0_s_uv

    tp_mask = f0_r_v * f0_s_v
    tn_mask = f0_r_uv * f0_s_uv
    fp_mask = f0_r_uv * f0_s_v
    fn_mask = f0_r_v * f0_s_uv

    if tone_shift is not None:
        shift_scale = 2 ** (tone_shift / 12)
        f0_r = f0_r * shift_scale

    # only calculate f0 error for voiced frame
    y = 1200 * np.abs(np.log2(f0_r + f0_r_uv) - np.log2(f0_s + f0_s_uv))
    y = y * tp_mask
    # print(y.sum(), tp_mask.sum())
    f0_rmse_mean = y.sum() / tp_mask.sum()

    # only voiced/ unvoiced accuracy/precision
    vuv_precision = tp_mask.sum() / (tp_mask.sum() + fp_mask.sum())
    vuv_accuracy = (tp_mask.sum() + tn_mask.sum()) / len(y)

    return f0_rmse_mean, vuv_accuracy, vuv_precision


def eval_rmse_ap(x_r, x_s, sr, frame_len='5'):
    # TODO: find out what algorithm to use.  maybe pyworld d4c?
    pass


if __name__ == '__main__':

    file_r = 'demo/exmaple_data/ground_truth/arctic_b0436.wav'
    file_s = 'demo/exmaple_data/no_pulse/arctic_b0436.wav'

    aud_r, sr_r = librosa.load(file_r, sr=None)
    aud_s, sr_s = librosa.load(file_s, sr=None)

    assert sr_r == sr_s
    if len(aud_r) != len(aud_s):
        aud_r = aud_r[:len(aud_s)]
        aud_s = aud_s[:len(aud_r)]

    # mcd = eval_MCD(aud_r, aud_s)
    rmse_f0 = eval_rmse_f0(aud_r, aud_s, sr_r)
    print(rmse_f0)

    # print(aud_r.shape)
    # print(eval_snr(aud_r, aud_s))
    # print(eval_snr(aud_r*10, aud_s*10))
