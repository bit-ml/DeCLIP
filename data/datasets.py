import os
import random
import pickle
from io import BytesIO
from PIL import Image, ImageOps, ImageFile
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from random import shuffle

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants for mean and standard deviation
MEAN = [0.48145466, 0.4578275, 0.40821073]

STD = [0.26862954, 0.26130258, 0.27577711]

# Helper functions
def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", 'tif', 'tiff']):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if file.split('.')[-1] in exts and must_contain in os.path.join(r, file):
                out.append(os.path.join(r, file))
    return out

def get_list(path, must_contain=''):
    if path.endswith(".pickle"):
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        return [item for item in image_list if must_contain in item]
    return recursively_read(path, must_contain)

def randomJPEGcompression(image):
    qf = random.randint(30, 100)
    output_io_stream = BytesIO()
    image.save(output_io_stream, "JPEG", quality=qf, optimize=True)
    output_io_stream.seek(0)
    return Image.open(output_io_stream)

# Base Dataset class
class BaseDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self._init_data()

    def _init_data(self):
        pass

    def _get_data(self):
        pass

    def _get_transform(self):
        transform_list = [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
        if self.opt.data_label == 'train':
            if self.opt.data_aug == "blur":
                transform_list.insert(1, transforms.GaussianBlur(kernel_size=5, sigma=(0.4, 2.0)))
            elif self.opt.data_aug == "color_jitter":
                transform_list.insert(1, transforms.ColorJitter(0.3, 0.3, 0.3, 0.3))
            elif self.opt.data_aug == "jpeg_compression":
                transform_list.insert(1, transforms.Lambda(randomJPEGcompression))
            elif self.opt.data_aug == "all":
                transform_list.insert(1, transforms.ColorJitter(0.3, 0.3, 0.3, 0.3))
                transform_list.insert(2, transforms.Lambda(randomJPEGcompression))
                transform_list.insert(3, transforms.GaussianBlur(kernel_size=5, sigma=(0.4, 2.0)))
        return transforms.Compose(transform_list)

    def __len__(self):
        pass

class RealFakeDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.mask_transf = self._get_mask_transform()

    def _init_data(self):
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
            self.masks_path = self.opt.train_masks_ground_truth_path
        elif self.opt.data_label == "valid":
            self.input_path = self.opt.valid_path
            self.masks_path = self.opt.valid_masks_ground_truth_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path
            self.masks_path = self.opt.test_masks_ground_truth_path

        fake_list = self._get_data()
        
        self.labels_dict = self._set_labels(fake_list)
        self.fake_list = fake_list
        shuffle(self.fake_list)
        self.transform = self._get_transform()

    def _get_data(self):
        fake_list = get_list(self.input_path)
                
        return fake_list

    def _get_mask_transform(self):
        return transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        
    def get_mask_from_file(self, file_name):
        if "autosplice" in self.opt.train_dataset:
            file_name = file_name[:file_name.rfind('_')] + "_mask.png"
        self.mask_path = os.path.join(self.masks_path, file_name)
        mask = Image.open(self.mask_path).convert("L")
        if self.opt.train_dataset in ['pluralistic', 'lama', 'repaint-p2-9k', 'ldm', 'ldm_clean', 'ldm_real']:
            mask = ImageOps.invert(mask)
        return self.mask_transf(mask).view(-1)

    def _set_labels(self, fake_list):
        # masks images should be .png
        labels = {img: img.split("/")[-1].replace(".jpg", ".png") for img in fake_list}
        return labels
    
    def __len__(self):
        return len(self.fake_list)

    def __getitem__(self, idx):
        img_path = self.fake_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.get_mask_from_file(label)

        return img, label, img_path, self.mask_path

class RealFakeDetectionDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)

    def _get_data(self):
        fake_list = get_list(self.input_path)
        real_list = get_list(self.input_path_real)
                
        return real_list, fake_list

    def _init_data(self):
        if self.opt.data_label == "train":
            self.input_path = self.opt.train_path
            self.input_path_real = self.opt.train_real_list_path
            self.masks_path = self.opt.train_masks_ground_truth_path
        elif self.opt.data_label == "valid":
            self.input_path = self.opt.valid_path
            self.input_path_real = self.opt.valid_real_list_path
            self.masks_path = self.opt.valid_masks_ground_truth_path
        elif self.opt.data_label == "test":
            self.input_path = self.opt.test_path
            self.input_path_real = self.opt.test_real_list_path
            self.masks_path = self.opt.test_masks_ground_truth_path

        real_list, fake_list = self._get_data()
        self.labels_dict = self._set_labels(real_list, fake_list)
        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        self.transform = self._get_transform()

    def _set_labels(self, real_list, fake_list):
        labels = {img: 0 for img in real_list}
        labels.update({img: 1 for img in fake_list})
        return labels
    
    def __len__(self):
        return len(self.total_list)
    
    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label, img_path