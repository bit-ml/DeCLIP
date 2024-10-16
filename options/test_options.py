from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ckpt', type=str, help='path to the trained model\'s checkpoint')
        parser.add_argument('--result_folder', type=str, default='result', help='path to the folder to log the test resutls')
        parser.add_argument('--output_save_path', type=str, default=None, help="The path to which the resulted images will be saved, along side the scores for each input sample")  
        
        # TODO: uncomment these line for backwards compability
        parser.add_argument('--decoder_type', type=str, default='conv-20', help='type of decoder (linear/attention/conv-4/conv-12/conv-20)')
        parser.add_argument('--feature_layer', type=str, default=None, help='layer of the backbone from which to extract features')
        self.data_label = 'test'
        return parser
