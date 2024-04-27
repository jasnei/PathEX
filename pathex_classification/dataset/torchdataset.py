import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.utils.data as Data
from utils.utils import default_loader, default_loader_cv2

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str, 
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]]=IMG_EXTENSIONS,
    is_valid_file: Optional[Callable[[str], bool]]=None,
    ) -> List[Tuple[str, int]]:
    instance = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError('Both extensions and is_valid_file cannot be None or not None at the same time')
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in fnames:
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = [path, class_index]
                    instance.append(item)

    return instance


def get_image_from_folder(target_dir: str, fn=is_image_file) -> List:
    """
    Description: get images from folder and return list
    """
    instances = []
    for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
        for fname in fnames:
            path = os.path.join(root, fname)
            if fn(path):
                # yield path
                instances.append(path)
    return instances
    

def single_load_transform(loader, transform, image):
    try:
        img = loader(image)
    except:
        print(image)
    if img is None:
        print(image)

    if transform is not None:
        augmented = transform(image=img)
        # img = Image.fromarray(augmented['image'])
        img = augmented['image']
    return img


def multi_load_transform(loader, num_workers, transform, files):
        images = []
        executor = ThreadPoolExecutor(max_workers=num_workers)
        partial_func = partial(single_load_transform, loader, transform)
        results = executor.map(partial_func, files)
        executor.shutdown()
        for result in results:
            images.append(result)
        ## results is iterator
        # images = [result for result in results]
        return images


class SimpleDataset(Data.Dataset):
    def __init__(self, imgs, transform=None, target_transform=None, loader=default_loader):
        if isinstance(imgs, np.ndarray):
            imgs = imgs.tolist()
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
    

class SimpleDatasetFn(Data.Dataset):
    def __init__(self, imgs, transform=None, target_transform=None, loader=default_loader):
        if isinstance(imgs, np.ndarray):
            imgs = imgs.tolist()
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)

        return fn, img, label

    def __len__(self):
        return len(self.imgs)


class SimpleDatasetFnAlbum(Data.Dataset):
    def __init__(self, imgs, transform=None, target_transform=None, loader=default_loader_cv2):
        if isinstance(imgs, np.ndarray):
            imgs = imgs.tolist()
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(image=img)['image']

        return fn, img, label

    def __len__(self):
        return len(self.imgs)

    
class TestDataset(Data.Dataset):
    def __init__(self, imgs, transform=None, target_transform=None, loader=default_loader_cv2):
        if isinstance(imgs, np.ndarray):
            imgs = imgs.tolist()
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, label

    def __len__(self):
        return len(self.imgs)


class WSITestDataset(TestDataset):
    def __init__(self, imgs, transform=None, target_transform=None, loader=None):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        deep_gen, deep_level, patch_name, j, i = self.imgs[index]
        patch = deep_gen.get_tile(deep_level, (j, i))
        img = np.asarray(patch)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return patch_name, img, patch


class DatasetFolder(Data.Dataset):
    """
    Description:
        - this could get the dataset from directory, you could pass a transform (not two,
          one for train, one for valid), so you might want to  get seperate
    """
    def __init__(self, 
                root: str, 
                batch_num, 
                transform=None, 
                multi_load=True,
                shuffle=True,
                seed=None,
                drop_last=False,
                num_workers=4,
                loader=default_loader_cv2,
                is_valid_file=None,
                percentage=1) -> None:

        super().__init__()

        classes, class_to_idx = find_classes(root)
        self.samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS, is_valid_file)
        print(class_to_idx, classes, len(self.samples))

        if len(self.samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.batch_num = batch_num
        self.transform = transform
        self.multi_load = multi_load
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.loader = loader

        self.multi_load_transform = partial(multi_load_transform, self.loader, 
                                            self.num_workers, self.transform)
        self.single_load_transform = partial(single_load_transform, self.loader, 
                                             self.transform)

        #==============================================
        # Set seed
        #==============================================
        # if seed is None:
        #     self.seed = random.randint(0, 1e6)
        # else:
        #     self.seed = seed
        # random.seed(self.seed)

        self.batches = self._create_batches()
        # print(len(self.batches))
        self.batches = self._get_len_batches(percentage)
        # print(type(self.batches))

    def _create_batches(self,):
        if self.shuffle:
            random.shuffle(self.samples)

        batches = []
        ranges = list(range(0, len(self.samples), self.batch_num))
        for i in ranges[:-1]:
            batch = self.samples[i:i + self.batch_num]
            batches.append(batch)

        #== Drop last ===============================================
        last_batch = self.samples[ranges[-1]:]
        if len(last_batch) == self.batch_num:
            batches.append(last_batch)
        elif self.drop_last:
            pass
        else:
            batches.append(last_batch)

        return batches

    def _get_len_batches(self, percentage):
        """
        Description:
            - you could control how many bags you want to use for training or validating
              indices sort, so that could keep the bags got in order from originla bags
        
        Parameters:
            - percentage: float, range [0, 1]

        Return
            - numpy array of the new bags
        """
        batch_num = int(len(self.batches) * percentage)
        indices = random.sample(list(range(len(self.batches))), batch_num)
        indices.sort()
        new_batches = np.array(self.batches, dtype='object')[indices]
        
        return new_batches

    def __getitem__(self, index):
        batch = np.array(self.batches[index])
        image_paths = batch[:, 0]
        labels = batch[:, 1].astype(np.uint8)

        ## Stack all images, become a 4 dimensional tensor
        if self.multi_load:
            batch_images = self.multi_load_transform(image_paths)

        else:
            batch_images = []
            for image in image_paths:
                img = self.single_load_transform(image)
                batch_images.append(img)

        batch_images_tensor = torch.stack(batch_images, dim=0)
        labels = torch.LongTensor(labels)

        return batch_images_tensor, labels

    def __len__(self):
        return len(self.batches)


class ListDataset(Data.Dataset):
    """
    Description:
        - this is the dataset you could pass a list of your own data
    """
    def __init__(self, 
                images, 
                batch_num, 
                percentage=1,
                transform=None, 
                multi_load=True,
                shuffle=True,
                seed=None,
                drop_last=False,
                num_workers=4,
                loader=default_loader_cv2) -> None:
        
        #==============================================
        # Set seed
        #==============================================
        # if seed is None:
        #     self.seed = np.random.randint(0, 1e6, 1)[0]
        # else:
        #     self.seed = seed
        # random.seed(self.seed)

        self.images = images
        self.batch_num = batch_num
        self.percentage = percentage
        self.transform = transform
        self.multi_load = multi_load
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.loader = loader

        self.multi_load_transform = partial(multi_load_transform, self.loader, 
                                            self.num_workers, self.transform)
        self.single_load_transform = partial(single_load_transform, self.loader, 
                                             self.transform)
        

        self.batches = self._create_batches()
        self.batches = self._get_len_batches(self.percentage)

    def _get_len_batches(self, percentage):
        """
        Description:
            - you could control how many bags you want to use for training or validating
              indices sort, so that could keep the bags got in order from originla bags
        
        Parameters:
            - percentage: float, range [0, 1]

        Return
            - numpy array of the new bags
        """
        batch_num = int(len(self.batches) * percentage)
        indices = random.sample(list(range(len(self.batches))), batch_num)
        indices.sort()
        new_batches = np.array(self.batches, dtype='object')[indices]
        return new_batches

    def _create_batches(self,):
        if self.shuffle:
            # random.shuffle(self.images)
            np.random.shuffle(self.images)

        batches = []
        ranges = list(range(0, len(self.images), self.batch_num))
        for i in ranges[:-1]:
            batch = self.images[i:i + self.batch_num]
            batches.append(batch)

        #== Drop last ===============================================
        last_batch = self.images[ranges[-1]:]
        if len(last_batch) == self.batch_num:
            batches.append(last_batch)
        elif self.drop_last:
            pass
        else:
            batches.append(last_batch)

        return batches

    def __getitem__(self, index):
        batch = np.array(self.batches[index])
        image_paths = batch[:, 0]
        labels = batch[:, 1].astype(np.uint8)

        ## Stack all images, become a 4 dimensional tensor
        if self.multi_load:
            batch_images = self.multi_load_transform(image_paths)

        else:
            batch_images = []
            for image in image_paths:
                img = self.single_load_transform(image)
                batch_images.append(img)

        batch_images_tensor = torch.stack(batch_images, dim=0)
        labels = torch.LongTensor(labels)

        return batch_images_tensor, labels

    def _load_transform(self, tile):
        try:
            img = self.loader(tile)
        except:
            print(tile)
        if img is None:
            print(tile)

        if self.transform is not None:
            augmented = self.transform(image=img)
            # img = Image.fromarray(augmented['image'])
            img = augmented['image']
        return img

    def _multi_loader(self, tiles):
        images = []
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        results = executor.map(self._load_transform, tiles)
        executor.shutdown()
        for result in results:
            images.append(result)
        ## results is iterator
        # images = [result for result in results]
        return images

    def __len__(self):
        return len(self.batches)


def train_test_split(dataset, train_ratio=0.7, shuffle=True):
    """
    np.random.seed(seed)
    """
    # train_test split
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    if shuffle:
        indices = np.random.permutation(dataset_size)
    split = int(np.ceil(train_ratio * dataset_size))
    train_indices, valid_indices = indices[:split], indices[split:]
    
    return dataset[train_indices], dataset[valid_indices]


def train_test_split_class(dataset, train_ratio=0.7, shuffle=True):
    """
    train test split dataset, consider split each class into two sets.
    
    Args
        - dataset: (numpy.array), dataset is numpy array, dtype is object, each row
                    is [path, class_index] like.
        - train_ratio: (float), tran ratio
        - shuffle: (boolen), whether shuffle the split set or not
        
    Returns:
        - tuple: (train, valid), train and valid are numpy.array, not index
    """
    train = []
    valid = []
    
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset, dtype=object)
    
    uniques = np.unique(dataset[:, 1])

    for unique in uniques:
        unique_samples = dataset[dataset[:, 1] == uniques[unique]]

        size_unique = len(unique_samples)
        indices = np.arange(size_unique)
        if shuffle:
            indices = np.random.permutation(size_unique)
        split = int(np.ceil(train_ratio * size_unique))
        train_indices, valid_indices = indices[:split], indices[split:]
        train.append(unique_samples[train_indices])
        valid.append(unique_samples[valid_indices])
    train = np.concatenate(train)
    valid = np.concatenate(valid)
    if shuffle:
        np.random.shuffle(train)
        np.random.shuffle(valid)
    
    return train, valid


if __name__ == '__main__':
    import time

    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader, SubsetRandomSampler
    image_size = 224
    root = os.path.abspath('D:/Data_set/flower_photos/')

    # Seperate dataset could to train valid and pass each transform
    classes, class_to_idx = find_classes(root)
    samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
    samples = np.asarray(samples, dtype=object)
    print(len(samples))

    train_indices, valid_indices = train_test_split(samples, train_ratio=0.7, shuffle=True, seed=42)

    train_samples = samples[train_indices]
    vadlid_samples = samples[valid_indices]

    albumentations_valid = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.1125, 0.1125, 0.1125,], std=[0.2077, 0.2077, 0.2077,]),
        ToTensorV2(),
    ])

    train_dataset = ListDataset(train_samples, 64, transform=albumentations_valid)
    train_loader = DataLoader(train_dataset, batch_size=1)
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        print(images.squeeze(0).size(), labels.squeeze(0).size())
    print(f'Comsun: {time.time() - start_time}')
    

    # # ==============================================================================================
    # # Folder dataset, 
    # # ==============================================================================================

    # batch_num = 64
    # dataset = DatasetFolder(root, batch_num, transform=albumentations_valid, drop_last=False, seed=42, multi_load=False)
    
    # # ===================================================================================================
    # # train_test_split method 1
    # # ===================================================================================================
    # # train_indices, valid_indices = train_test_split(dataset, train_ratio=0.7, shuffle=True, seed=None)

    # # train_sampler = SubsetRandomSampler(train_indices)
    # # valid_sampler = SubsetRandomSampler(valid_indices)
    
    # # train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
    # # valid_loader = DataLoader(dataset, batch_size=1, sampler=valid_sampler)

    # # ===================================================================================================
    # # train_test_split method 2
    # # ===================================================================================================
    # torch.random.manual_seed(42)
    # ratio = 0.8
    # dataset_size = len(dataset)
    # # print(dataset_size)
    # train_size = int(np.ceil(ratio * dataset_size))
    # valid_size = len(dataset) - train_size
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    # train_loader = DataLoader(train_dataset, batch_size=1, drop_last=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, drop_last=False)
    # # print(len(train_loader), len(valid_loader))
    # start_time = time.time()
    # for i, (images, labels) in enumerate(train_loader):
    #     print(images.squeeze(0).size(), labels.squeeze(0).size())
    # print(f'Comsun: {time.time() - start_time}')
