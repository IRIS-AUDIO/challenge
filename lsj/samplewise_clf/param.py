import argparse


def get_args(known = []):
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, required=True)
    args.add_argument('--model', type=str, default='EfficientNetB0')
    args.add_argument('--mode', type=str, default='regr', choices=['regr', 'clf'])
    args.add_argument('--pretrain', type=bool, default=False)
    # args.add_argument('--framewise', action='store_true')

    # Model
    args.add_argument('--feature_extractor', type=str, default='EfficientNetB0', choices=['EfficientNetB'+str(i) for i in range(8)])
    args.add_argument('--predict', type=str, default='sample', choices=['sample', 'frame'])
    abspath = '/root/otherperson/daniel'
    # DATA
    args.add_argument('--background_sounds', type=str,
                    default=abspath + '/generate_wavs/drone_normed_complex.pickle')
    args.add_argument('--voices', type=str,
                    default=abspath + '/generate_wavs/voice_normed_complex.pickle')
    args.add_argument('--labels', type=str,
                    default=abspath + '/generate_wavs/voice_labels_mfc.npy')
    args.add_argument('--noises', type=str,
                    default=abspath + '/RDChallenge/tf_codes/sounds/noises_specs.pickle')

    # TRAINING
    args.add_argument('--optimizer', type=str, default='adam',
                                    choices=['adam', 'sgd', 'rmsprop'])
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--lr_factor', type=float, default=0.7)
    args.add_argument('--lr_patience', type=int, default=10)

    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--n_frame', type=int, default=1000)
    args.add_argument('--steps_per_epoch', type=int, default=500)
    args.add_argument('--l2', type=float, default=1e-6)
    args.add_argument('--n_dim', type=int, default=256)

    # TEST
    args.add_argument('--test_background_sounds', type=str,
                    default=abspath + '/generate_wavs/test_drone_normed_complex.pickle')
    args.add_argument('--test_voices', type=str,
                    default=abspath + '/generate_wavs/test_voice_normed_complex.pickle')
    args.add_argument('--test_labels', type=str,
                    default=abspath + '/generate_wavs/test_voice_labels_mfc.npy')
    args.add_argument('--n_mels', type=int, default=100)
    args.add_argument('--n_classes', type=int, default=30)

    # AUGMENTATION
    args.add_argument('--alpha', type=float, default=0.75)
    args.add_argument('--snr', type=float, default=-10)
    args.add_argument('--max_voices', type=int, default=4)
    args.add_argument('--max_noises', type=int, default=2)

    return args.parse_known_args(known)[0]