from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--train_dataset', type=str, default='pluralistic', help='the dataset on which to train')
        parser.add_argument('--decoder_type', type=str, default='conv-20', help='type of decoder (linear/attention/conv-4/conv-12/conv-20)')
        parser.add_argument('--feature_layer', type=str, default=None, help='layer of the backbone from which to extract features')
        parser.add_argument('--data_aug', type=str, default=None, help='if specified, perform additional data augmentation (blur/color_jitter/jpeg_compression/all)')
        
        parser.add_argument('--earlystop_epoch', type=int, default=5)
        parser.add_argument('--optim', type=str, default='adam', help='optim to use (sgd/adam)')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

        parser.add_argument('--loss_freq', type=int, default=50, help='frequency of showing loss on tensorboard')
        parser.add_argument('--niter', type=int, default=400, help='total epochs')
        

        parser.add_argument('--data_root_path', type=str, default=None, help='Root path for dolos data only! Explicitly fill in the other data paths for other dataset.')
        parser.add_argument('--train_path', type=str, default='datasets/dolos_data/celebahq/fake/ldm/images/train', help='folder path to training fake data')
        parser.add_argument('--valid_path', type=str, default='datasets/dolos_data/celebahq/fake/ldm/images/valid', help='folder path to validation fake data')
        parser.add_argument('--train_masks_ground_truth_path', type=str, default='datasets/dolos_data/celebahq/fake/ldm/masks/train', help='path to train ground truth masks (only for fully_supervised training)')
        parser.add_argument('--valid_masks_ground_truth_path', type=str, default='datasets/dolos_data/celebahq/fake/ldm/masks/valid', help='path to valid ground truth masks (only for fully_supervised training)')
        parser.add_argument('--train_real_list_path', default='datasets/dolos_data/celebahq/real/train', help='folder path to training real data')
        parser.add_argument('--valid_real_list_path', default='datasets/dolos_data/celebahq/real/valid', help='folder path to validation real data')        
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')

        self.data_label = 'train'
        return parser
