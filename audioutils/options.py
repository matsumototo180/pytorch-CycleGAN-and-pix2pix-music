import argparse
import os
from pathlib import Path

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()    # パーサを作る
        self.parser.add_argument('--input', required=True, help='path to audio file or directory')
        # additional parameters
        self.parser.add_argument('--output', default='./', help='path to output directory')
        self.parser.add_argument('--sr', type=int, default='22050', help='sampling rate')
        self.parser.add_argument('--stereo', action='store_true', help='if specified, load audio as stereo source')
        self.parser.add_argument('--wl', type=int, default='1024', help='window length')
        self.parser.add_argument('--hl', type=int, default='512', help='hop length')
        self.parser.add_argument('--cl', type=int, default='128', help='signal crop length')

    def parse(self):
        self.opt = self.parser.parse_args() 
        self.opt.input = Path(self.opt.input)
        self.opt.output = Path(self.opt.output)
        return self.opt
    
    def parse_test(self):
        self.opt = self.parser.parse_args(args=["--input", "/home/matsumoto/Downloads/fma/fma_small/000/", "--output", "./Rock"])
        # self.opt = self.parser.parse_args(args=["--input", "/home/matsumoto/Downloads/fma/fma_small_separated/Rock/"])
        # self.opt = self.parser.parse_args(args=["--input", "/home/matsumoto/Downloads/fma/fma_small_separated/Rock/Rock.066536.wav"])
        # self.opt = self.parser.parse_args() 
        self.opt.input = Path(self.opt.input)
        self.opt.output = Path(self.opt.output)
        return self.opt


    # def gather_options(self):
    #     if not self.initialized:  # check if it has been initialized
    #         parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #         parser = self.initialize(parser)

    #     # get the basic options
    #     opt, _ = parser.parse_known_args()

    #     # modify model-related parser options
    #     model_name = opt.model
    #     model_option_setter = models.get_option_setter(model_name)
    #     parser = model_option_setter(parser, self.isTrain)
    #     opt, _ = parser.parse_known_args()  # parse again with new defaults

    #     # modify dataset-related parser options
    #     dataset_name = opt.dataset_mode
    #     dataset_option_setter = data.get_option_setter(dataset_name)
    #     parser = dataset_option_setter(parser, self.isTrain)

    #     # save and return the parser
    #     self.parser = parser
    #     return parser.parse_args()

    #     def parse(self):    
    #         opt = self.gather_options()
    #         opt.isTrain = self.isTrain   # train or test

    #     # process opt.suffix
    #     if opt.suffix:
    #         suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
    #         opt.name = opt.name + suffix

    #     self.print_options(opt)

    #     # set gpu ids
    #     str_ids = opt.gpu_ids.split(',')
    #     opt.gpu_ids = []
    #     for str_id in str_ids:
    #         id = int(str_id)
    #         if id >= 0:
    #             opt.gpu_ids.append(id)
    #     if len(opt.gpu_ids) > 0:
    #         torch.cuda.set_device(opt.gpu_ids[0])

    #     self.opt = opt
    #     return self.opt
