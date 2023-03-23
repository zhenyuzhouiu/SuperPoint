import numpy as np
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import experiment
from superpoint.settings import EXPER_PATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/fingernail-minutiae_fingernail_export.yaml')
    parser.add_argument('--experiment_name', type=str, default='fingernail-minutiae_fingernail_multihead')
    parser.add_argument('--export_name', type=str, default='fingernail-minutiae-multihead_export')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--pred_only', action='store_true', default=True)  # default value False
    parser.add_argument('--head_mode', type=str, default='multiple')
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = args.export_name if args.export_name else experiment_name
    batch_size = args.batch_size
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    assert 'eval_iter' in config

    output_dir = Path(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    config['model']['pred_batch_size'] = batch_size
    batch_size *= experiment.get_num_gpus()

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load(str(checkpoint))
        test_set = dataset.get_test_set()

        for _ in tqdm(range(config.get('skip', 0))):
            next(test_set)

        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        while True:
            # Gather dataset
            data = []
            try:
                # get a batch size of test data
                for _ in range(batch_size):
                    data.append(next(test_set))
            except (StopIteration, dataset.end_set):
                if not data:
                    break
                data += [data[-1] for _ in range(batch_size - len(data))]  # add dummy
            # zip(*) can be regard unzipped
            # dict{key:data[0], values:d.values()}
            # zip(dict, tuple) will only use the keys on the dict
            # data = {'image': tuple(batch), 'name': tuple(batch)}
            data = dict(zip(data[0], zip(*[d.values() for d in data])))  # list to dictionary

            # Predict
            # for minutiae_head keys: {'minutiae_raw': x, 'prob': prob, 'classes': cls,
            # 'angles': ang, 'prob_nms': prob, 'pred': pred}

            # for multiple head keys: {'logits': x, 'prob': prob, 'classes_raw': x, 'classes': cls,
            # 'angles_raw': x, 'angles': ang, 'prob_nms': prob, 'pred': pred}
            if args.pred_only:
                if args.head_mode == 'minutiae':
                    # output p is an array with [N, H, W]
                    p = net.predict(data, keys='pred', batch=True)
                    # for e in p for iterating batch size
                    # pred = {'points': [(n1,2), (n2,2), (n3,2)]} # from the keypoint map to keypoint position
                    # np.argwhere == np.array(np.where(e)).T
                    pred = {'points': [np.array(np.where(e)).T for e in p]}  # (row, column)
                else:
                    # output p is an array with [N, H, W]
                    p = net.predict(data, keys=['pred', 'classes', 'angles'], batch=True)
                    p_p = p['pred']
                    p_c = p['classes']
                    p_a = p['angles']
                    pred_p, pred_c, pred_a = [], [], []
                    for b in range(batch_size):
                        p_p_i, p_c_i, p_a_i = p_p[b], p_c[b], p_a[b]
                        row_index, column_index = np.where(p_p_i)
                        pred_p.append(np.array((row_index, column_index)).T)
                        pred_c.append(p_c_i[row_index, column_index].reshape(-1, 1))
                        pred_a.append(p_a_i[row_index, column_index].reshape(-1, 1))

                    pred = {'points': pred_p, 'classes': pred_c, 'angles': pred_a}
            else:
                # pred is a list
                # pred = {'prob': (batch, h, w), 'counts': (batch, h, w), 'mean_prob': (batch, h, w) ...}
                pred = net.predict(data, keys='*', batch=True)

            # Export
            # dictionary to list
            d2l = lambda d: [dict(zip(d, e)) for e in zip(*d.values())]  # noqa: E731  dictionary to list
            for p, d in zip(d2l(pred), d2l(data)):
                if not ('name' in d):
                    p.update(d)  # Can't get the data back from the filename --> dump
                filename = d['name'].decode('utf-8') if 'name' in d else str(i)
                filepath = Path(output_dir, '{}.npz'.format(filename))
                np.savez_compressed(filepath, **p)
                i += 1
                pbar.update(1)

            if config['eval_iter'] > 0 and i >= config['eval_iter']:
                break
