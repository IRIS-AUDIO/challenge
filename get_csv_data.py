import csv
from glob import glob
import os
from numpy import max, mean

from tqdm import tqdm

from sj_train import ARGS, get_model
from eval import evaluate


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    data_path = config.path
    paths = sorted(glob(os.path.join(data_path, '*.csv')))
    result_path = os.path.join(data_path, 'result.csv')
    category = ['이름', '모델', 'version', 'batch', 'lr', 'optimizer', 'loss function', 'input', 'chan', 'output', 'epoch', 'cos_sim', 'er', 'f1_score', 'loss', 'precision', 'val_cos_sim', 'val_er', 'val_f1_score', 'val_loss', 'val_precision', 'test_er', 'swa_test_er']

    prev_lines = [category]
    
    if len(prev_lines) == 0:
        with open(result_path, 'w') as f:
            wr = csv.writer(f)
            wr.writerow(category)

    for path in tqdm(paths):
        if path == result_path:
            continue

        lines = []
        with open(path, 'r') as f:
            data = csv.reader(f)
            for i, line in enumerate(data):
                if i == 0:
                    continue
                lines.append(line)
        data = lines[max([len(lines)-config.patience, 0])]
        filename = os.path.splitext(path.split('/')[-1])[0]
        name = filename[filename.find('B'):].split('_')
        model_name = name[0]
        version = name[1][1:]
        lr = name[2][2:]
        batch = name[3].split('batch')[-1]
        opt = name[5]
        n_mel = name[6].split('mel')[-1]
        chan = name[7].split('chan')[-1]
        loss = name[8]
        framelen = name[9].split('framelen')[-1]
        evaluation = max([len(lines)-config.patience, 0]) > 5

        
        config.model = model_name[1:]
        config.v = int(version)
        config.n_mels = int(n_mel)
        config.n_chan = int(chan)
        config.n_frame = int(framelen)

        model = get_model(config)
        data = [filename, model_name, version, batch, lr, opt, loss, str(tuple([i for i in model.input.shape[1:-1]])), chan, str(tuple([i for i in model.output.shape[1:]]))] + data
        if os.path.exists(f'{os.path.splitext(path)[0]}.h5'):
            if evaluation:
                model.load_weights(f'{os.path.splitext(path)[0]}.h5')
                score = evaluate(config, model, overlap_hop=int(framelen) // 2, verbose=True)
            else:
                score = 1.0
            data += [mean(score)]
        else:
            data += 'None'

        if os.path.exists(f'{os.path.splitext(path)[0]}_SWA.h5'):
            if evaluation:
                model.load_weights(f'{os.path.splitext(path)[0]}_SWA.h5')
                score = evaluate(config, model, overlap_hop=int(framelen) // 2, verbose=True)
            else:
                score = 1.0
            data += [mean(score)]
        else:
            data += ['None']
        prev_lines.append(data)

    with open(result_path, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(prev_lines)


if __name__ == '__main__':
    args = ARGS()
    args.args.add_argument('--path', type=str, default='')
    main(args.get())
