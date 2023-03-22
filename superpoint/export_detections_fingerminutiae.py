import numpy as np
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pred_only', action='store_true')  # default value False
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
            data = dict(zip(data[0], zip(*[d.values() for d in data])))

            # Predict
            # for minutiae_head keys: {'minutiae_raw': x, 'prob': prob, 'c_prob': c_prob,
            # 'a_prob': a_prob, 'prob_nms': prob, 'pred': pred}

            # for multiple head keys: {'logits': x, 'prob': prob, 'classes_raw': x, 'classes': cls,
            # 'angles_raw': x, 'angles': ang, 'prob_nms': prob, 'pred': pred}
            if args.pred_only:
                p = net.predict(data, keys='pred', batch=True)
                pred = {'points': [np.array(np.where(e)).T for e in p]}  # from the keypoint map to keypoint position
            else:
                pred = net.predict(data, keys='*', batch=True)

            # Export
            # dictionary to list
            d2l = lambda d: [dict(zip(d, e)) for e in zip(*d.values())]  # noqa: E731
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
