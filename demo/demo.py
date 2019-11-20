from vocoder_eva.eval import eval_MCD, eval_rmse_f0, eval_snr
import os
from glob import glob
from pathlib import Path
import librosa


class EvaDataset:

    def __init__(self, raw_folder, syn_folder):
        self.r_files = {Path(y).stem: Path(y) for x in os.walk(raw_folder) for y in glob(os.path.join(x[0], '*.wav'))}
        self.s_files = {Path(y).stem: Path(y) for x in os.walk(syn_folder) for y in glob(os.path.join(x[0], '*.wav'))}

        assert len(self.r_files) == len(self.r_files)

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


if __name__ == '__main__':
    r_folder = '../data/ground_truth'
    s_folder = '../data/repeat1_pulse'

    d = EvaDataset(r_folder, s_folder)

    f0_rmse_list = []
    vuv_precision_list = []

    for aud_r, aud_s, sr in d:
        f0_rmse_mean, vuv_precision = eval_rmse_f0(aud_r, aud_s, sr, method='dio')
        print(f0_rmse_mean, vuv_precision)
        f0_rmse_list.append(f0_rmse_mean)
        vuv_precision_list.append(vuv_precision)

    avg_f0_rmse = sum(f0_rmse_list) / len(d)
    avg_vuv_precision = sum(vuv_precision_list) / len(d)

    print('avg_f0_rmse:', avg_f0_rmse, 'avg_vuv_precision:', avg_vuv_precision)
