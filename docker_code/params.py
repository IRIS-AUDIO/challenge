import argparse


def getArgs(known = []):
    args = argparse.ArgumentParser()
    args.add_argument('path', type=str)
    args.add_argument('--model', type=str, default='EfficientNetB4')
    args.add_argument('--mode', type=str, default='GRU',
                                    choices=['GRU', 'transformer'])
    args.add_argument('--n_layers', type=int, default=0)
    args.add_argument('--n_dim', type=int, default=256)
    args.add_argument('--n_heads', type=int, default=8)

    args.add_argument('--n_mels', type=int, default=128)
    args.add_argument('--n_classes', type=int, default=30)

    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--n_frame', type=int, default=2048)

    # AUGMENTATION
    args.add_argument('--alpha', type=float, default=0.75)
    args.add_argument('--snr', type=float, default=-10)
    args.add_argument('--max_voices', type=int, default=6)
    args.add_argument('--max_noises', type=int, default=3)

    args.add_argument('--multiplier', type=float, default=10.)

    config = args.parse_known_args(known)[0]
    return config

if __name__ == "__main__":
    import sys
    print(getArgs(sys.argv[1:]))