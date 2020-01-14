from vocoder_eva.eval import eval_MCD, eval_rmse_f0, eval_snr, plot_f0
import os
from glob import glob
from pathlib import Path
import librosa


class EvaDataset:

    def __init__(self, raw_folder, syn_folder):
        self.r_files = {Path(y).stem: Path(y) for x in os.walk(raw_folder) for y in glob(os.path.join(x[0], '*.wav'))}
        self.s_files = {Path(y).stem: Path(y) for x in os.walk(syn_folder) for y in glob(os.path.join(x[0], '*.wav'))}

        self.s_files = {file_name: file_path for file_name, file_path in self.s_files.items() if
                        file_name.find('-') == -1}
        self.s_files = {file_name: file_path for file_name, file_path in self.s_files.items() if
                        file_name.find('+') == -1}

        assert len(self.r_files) == len(self.s_files)

        self.file_names = set(self.r_files.keys()).union(set(self.s_files.keys()))
        self.file_names = list(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        r_file = self.r_files[file_name]
        s_file = self.s_files[file_name]

        aud_r, sr_r = librosa.load(r_file, sr=None)
        aud_s, sr_s = librosa.load(s_file, sr=None)

        assert sr_r == sr_s

        return aud_r, aud_s, sr_r


def evaluate_f0(dataset: EvaDataset, tone_shift=0):
    f0_rmse_list = []
    vuv_precision_list = []

    for aud_r, aud_s, sr in dataset:
        f0_rmse_mean, vuv_accuracy, vuv_precision = eval_rmse_f0(aud_r, aud_s, sr, method='dio', tone_shift=tone_shift)
        # print(f0_rmse_mean, vuv_precision)
        f0_rmse_list.append(f0_rmse_mean)
        vuv_precision_list.append(vuv_precision)

    avg_f0_rmse = sum(f0_rmse_list) / len(dataset)
    avg_vuv_precision = sum(vuv_precision_list) / len(dataset)

    print('avg_f0_rmse:', avg_f0_rmse, 'avg_vuv_precision:', avg_vuv_precision)


if __name__ == '__main__':
    # r: raw, s: synthesised
    r_folder = '../data/ground_truth'
    s_folder_3 = '../data/repeat1_no_pulse/semi_tone_shift_repeat1+1'

    n0 = EvaDataset(r_folder, '../data/repeat1_no_pulse/repeat1_no_pulse')
    np1 = EvaDataset(r_folder, '../data/repeat1_no_pulse/semi_tone_shift_repeat1-1')
    nn1 = EvaDataset(r_folder, '../data/repeat1_no_pulse/semi_tone_shift_repeat1+1')

    p0 = EvaDataset(r_folder, '../data/out_shifts0113/repeat2_7layer_01130')
    pp1 = EvaDataset(r_folder, '../data/out_shifts0113/repeat2_7layer_01131')
    pn1 = EvaDataset(r_folder, '../data/out_shifts0113/repeat2_7layer_0113-1')

    # print('## case : no pulse ##')
    # evaluate_f0(n0)
    # print('## case : pulse ##')
    # evaluate_f0(p0)

    # print('## case : no pulse -1 ##')
    # evaluate_f0(nn1, tone_shift=-1)
    # print('## case : pulse -1##')
    # evaluate_f0(pn1, tone_shift=-1)
    #
    # print('## case : no pulse +1 ##')
    # evaluate_f0(np1, tone_shift=+1)
    # print('## case : pulse +1##')
    # evaluate_f0(pp1, tone_shift=+1)

    file_name = nn1.file_names[10]

    files = [n0.r_files[file_name],
             n0.s_files[file_name],
             nn1.s_files[file_name],
             # pn1.s_files[file_name],
             ]

    files2 = [
        nn1.s_files[file_name],
        # pn1.s_files[file_name],
    ]

    # nagoya baseline
    # files3 = [
    #     (n0.r_files[file_name], 'ground truth'),
    #     (n0.s_files[file_name], 'no shift'),
    #     (nn1.s_files[file_name], '+1 semitone'),
    #     (np1.s_files[file_name], '-1 semitone'),
    # ]
    # plot_f0(*files3, title='f0 curve : [Nagoya baseline wavenet]')

    files4 = [
        (p0.r_files[file_name], 'ground truth'),
        (p0.s_files[file_name], 'no shift'),
        (pp1.s_files[file_name], '+1 semitone'),
        (pn1.s_files[file_name], '-1 semitone')
    ]
    plot_f0(*files4, title='f0 curve : [F0-editable 7x2 layer 01/14]')

    # plot_f0(*files)
