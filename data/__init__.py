import torch
from .datasets import RealFakeDataset, RealFakeDetectionDataset


def create_dataloader(opt):
    shuffle = True if opt.data_label == 'train' else False
    if opt.fully_supervised:
        dataset = RealFakeDataset(opt)
    else:
        dataset = RealFakeDetectionDataset(opt)
    
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              num_workers=int(opt.num_threads))
    return data_loader
