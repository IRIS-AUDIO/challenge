import argparse


def get_args(known = []):
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, required=True)
    args.add_argument('--model', type=str, default='EfficientNetB4')
    args.add_argument('--pretrain', type=bool, default=False)
    args.add_argument('--framewise', type='store_true')

    # Model
    args.add_argument('--feature_extractor', type=str, default='EfficientNetB0', choi)

    # DATA
    args.add_argument('--background_sounds', type=str,
                    default='/codes/generate_wavs/drone_normed_complex.pickle')
    args.add_argument('--voices', type=str,
                    default='/codes/generate_wavs/voice_normed_complex.pickle')
    args.add_argument('--labels', type=str,
                    default='/codes/generate_wavs/voice_labels_mfc.npy')
    args.add_argument('--noises', type=str,
                    default='/codes/RDChallenge/tf_codes/sounds/noises_specs_2.pickle')

    # TRAINING
    args.add_argument('--optimizer', type=str, default='adam',
                                    choices=['adam', 'sgd', 'rmsprop'])
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--lr_factor', type=float, default=0.7)
    args.add_argument('--lr_patience', type=int, default=10)

    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--n_frame', type=int, default=1000)
    args.add_argument('--steps_per_epoch', type=int, default=500)
    args.add_argument('--l2', type=float, default=1e-6)

    # AUGMENTATION
    args.add_argument('--alpha', type=float, default=0.75)
    args.add_argument('--snr', type=float, default=-10)
    args.add_argument('--max_voices', type=int, default=4)
    args.add_argument('--max_noises', type=int, default=2)

return args.parse_known_args(known)[0]