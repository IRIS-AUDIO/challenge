import csv
from glob import glob
import os
from numpy import max, mean

from tqdm import tqdm

from sj_train import ARGS, get_model
from metrics import Challenge_Metric, er_score
from eval import evaluate


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    paths = sorted(glob('*.csv'))
    result_path = 'result.csv'
    category = ['이름', '모델', 'version', 'batch', 'lr', 'optimizer', 'input', 'chan', 'output', 'epoch', 'cos_sim', 'er', 'f1_score', 'loss', 'precision', 'val_cos_sim', 'val_er', 'val_f1_score', 'val_loss', 'val_precision', 'test_er', 'swa_test_er']
    data = [category]

    prev_lines = []
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            results = csv.reader(f)
            for result in results:
                prev_lines.append(result)
    
    if len(prev_lines) == 0:
        with open(result_path, 'w') as f:
            wr = csv.writer(f)
            wr.writerow(category)

    for path in tqdm(paths):
        if path == 'result.csv':
            continue

        lines = []
        with open(path, 'r') as f:
            data = csv.reader(f)
            for i, line in enumerate(data):
                if i == 0:
                    continue
                lines.append(line)
        data = lines[max([-config.patience, 0])]
        filename = os.path.splitext(path.split('/')[-1])[0]
        name = filename[filename.find('B'):].split('_')
        model = name[0]
        version = name[1][1:]
        lr = name[2][2:]
        batch = name[3].split('batch')[-1]
        opt = name[4]
        n_mel = name[6].split('mel')[-1]
        chan = name[7].split('chan')[-1]
        framelen = name[9].split('framelen')[-1]
        outputlen = f'{int(framelen) // 32 if int(version) == 3 else framelen}'
         
        data = [filename, model, version, batch, lr, opt, f'({n_mel}, {framelen})', chan, f'({outputlen}, 3)'] + data
        
        config.model = model[1:]
        config.v = int(version)
        config.n_mels = int(n_mel)
        config.n_chan = int(chan)

        model = get_model(config)
        metric = Challenge_Metric()
        if os.path.exists(f'{os.path.splitext(path)[0]}.h5'):
            model.load_weights(f'{os.path.splitext(path)[0]}.h5')
            score = evaluate(config, model, metric, verbose=True)
            data += [mean(score)]
        else:
            data += 'None'

        if os.path.exists(f'{os.path.splitext(path)[0]}_SWA.h5'):
            score = evaluate(config, model, metric, verbose=True)
            data += [mean(score)]
        else:
            data += ['None']
        prev_lines.append(data)

    with open(result_path, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(prev_lines)


if __name__ == '__main__':
    main(ARGS().get())
