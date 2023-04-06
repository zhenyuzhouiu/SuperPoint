# ===================
# Function: Merge the point, classes, and angle result to minutiae
# Author: ZHOU, Zhenyu
# Date: 2023.3.31
# ===================
import os.path
import shutil
import numpy as np


def merge(p_path, c_path, a_path, save_path):
    file_list = os.listdir(p_path)
    for f in file_list:
        p_npz = np.load(os.path.join(p_path, f))
        pred_p = p_npz['points']
        pred_prob = p_npz['prob']
        c_npz = np.load(os.path.join(c_path, f))
        pred_c = c_npz['classes']
        a_npz = np.load(os.path.join(a_path, f))
        pred_a = a_npz['angles']

        row_index, column_index = np.where(pred_p)
        p = np.array((row_index, column_index)).T
        c = np.array(pred_c[row_index, column_index].reshape(-1, 1))
        a = np.array(pred_a[row_index, column_index].reshape(-1, 1))
        prob = np.array(pred_prob[row_index, column_index].reshape(-1, 1))

        minutiae = {'points': p, 'classes': c, 'anlges': a, 'prob': prob}

        np.savez_compressed(os.path.join(save_path, f), **minutiae)


if __name__ == "__main__":
    p_path = '../exper_dir/outputs/fingernail-minutiae-p_fingernail/session2_crop'
    c_path = '../exper_dir/outputs/fingernail-minutiae-c_fingernail/session2_crop'
    a_path = '../exper_dir/outputs/fingernail-minutiae-a_fingernail/session2_crop'
    save_path = '../exper_dir/outputs/session2_crop'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    merge(p_path, c_path, a_path, save_path)

