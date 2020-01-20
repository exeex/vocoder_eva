from vocoder_eva.eval import eval_MCD, eval_rmse_f0, eval_snr
import os
from glob import glob
from pathlib import Path
import librosa


class WavDataset:

    def __init__(self, folder):
        self.files = {Path(y).stem: Path(y) for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.wav'))}

        # filter
        # self.files = {file_name: file_path for file_name, file_path in self.s_files.items() if
        #               file_name.find('-') == -1}
        # self.files = {file_name: file_path for file_name, file_path in self.s_files.items() if
        #               file_name.find('+') == -1}

        # self.file_names = set(self.r_files.keys()).union(set(self.s_files.keys()))
        self.file_names = set(self.files.keys())
        self.file_names = list(self.file_names)

    def without_str(self, match_string):
        self.files = {file_name: file_path for file_name, file_path in self.files.items() if
                      file_name.find(match_string) == -1}
        return self

    def include_str(self, match_string):
        self.files = {file_name: file_path for file_name, file_path in self.files.items() if
                      file_name.find(match_string) != -1}
        return self

    def replace_key(self, match_string, replace_str):
        self.files = {file_name.replace(match_string, replace_str): file_path for file_name, file_path in self.files.items()}
        return self


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, file_name):
        # file_name = self.file_names[idx]

        file = self.files[file_name]

        # aud_r, sr_r = librosa.load(file, sr=None)
        aud, sr = librosa.load(file, sr=None)

        # assert sr_r == sr_s

        return aud, sr


def common_files(a: WavDataset, b: WavDataset):
    _common_files = set(a.files.keys()).union(set(b.files.keys()))
    if not (len(a.files) == len(b.files) == len(_common_files)):
        print("[Warning] these 2 dataset not equal!!")
    return _common_files


def evaluate_f0(source: WavDataset, target: WavDataset, tone_shift=0):
    f0_rmse_list = []
    vuv_precision_list = []

    for file_name in common_files(source, target):
        aud_s, sr_s = source[file_name]
        aud_t, sr_t = target[file_name]
        assert sr_s == sr_t

        f0_rmse_mean, vuv_accuracy, vuv_precision = eval_rmse_f0(aud_s, aud_t, sr_s, method='dio',
                                                                 tone_shift=tone_shift)
        # print(f0_rmse_mean, vuv_precision)
        f0_rmse_list.append(f0_rmse_mean)
        vuv_precision_list.append(vuv_precision)

    avg_f0_rmse = sum(f0_rmse_list) / len(common_files(source, target))
    avg_vuv_precision = sum(vuv_precision_list) / len(common_files(source, target))

    print('avg_f0_rmse:', avg_f0_rmse, 'avg_vuv_precision:', avg_vuv_precision)


if __name__ == '__main__':
    # r: raw, s: synthesised
    r_folder = '../data/ground_truth'
    s_folder_3 = '../data/repeat1_no_pulse/semi_tone_shift_repeat1+1'

    gt = WavDataset(r_folder)
    bs_0 = WavDataset('../data/repeat1_no_pulse/repeat1_no_pulse')
    bs_1 = WavDataset('../data/repeat1_no_pulse/semi_tone_shift_repeat1-1')
    bs_n1 = WavDataset('../data/repeat1_no_pulse/semi_tone_shift_repeat1+1')

    # pu_0 = WavDataset('../data/out_shifts0113/repeat2_7layer_01130').without_str('-').without_str('+')
    # pu_1 = WavDataset('../data/out_shifts0113/repeat2_7layer_01131').without_str('-').without_str('+')
    # pu_n1 = WavDataset('../data/out_shifts0113/repeat2_7layer_0113-1').without_str('-').without_str('+')

    pu_0 = WavDataset('../data/eva_out_pulse0115_fff0').without_str('-').without_str('+')
    pu_1 = WavDataset('../data/eva_out_pulse0115_fff0').include_str('+').replace_key('+1', '')
    pu_n1 = WavDataset('../data/eva_out_pulse0115_fff0').include_str('-').replace_key('-1', '')

    print('## case : no pulse ##')
    evaluate_f0(gt, bs_0)
    print('## case : pulse ##')
    evaluate_f0(gt, pu_0)

    print('## case : no pulse -1 ##')
    evaluate_f0(gt, bs_n1, tone_shift=-1)
    print('## case : pulse -1##')
    evaluate_f0(gt, pu_n1, tone_shift=-1)

    print('## case : no pulse +1 ##')
    evaluate_f0(gt, bs_1, tone_shift=1)
    print('## case : pulse +1##')
    evaluate_f0(gt, pu_1, tone_shift=1)

    # a = [key for key in pu_n1.files.keys()]
    # a.sort()
    # print(a)
    #
    # a = [key for key in gt.files.keys()]
    # a.sort()
    # print(a)

    # for files in common_files(gt, pu_n1):
    #     print(files)