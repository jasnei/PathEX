# !/usr/bin/python
# -*- coding: utf-8 -*-
# jasnei@163.com
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import default_loader_cv2, default_loader


class ListDataset(Dataset):
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
        if seed is None:
            self.seed = np.random.randint(0, 1e6, 1)[0]
        else:
            self.seed = seed
        random.seed(self.seed)

        self.images = images
        self.batch_num = batch_num
        self.percentage = percentage
        self.transform = transform
        self.multi_load = multi_load
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.loader = loader

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
            random.shuffle(self.images)

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

        batch = self.batches[index]
        image_paths = batch[:, 0]
        labels = batch[:, 1].astype(np.uint8)

        ## Stack all images, become a 4 dimensional tensor
        if self.multi_load:
            batch_images = self._multi_loader(image_paths)

        else:
            batch_images = []
            for image in image_paths:
                img = self._load_transform(image)
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


if __name__ == '__main__':
    import time

    import albumentations as album
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader

    albumentations_valid = album.Compose([
        album.Resize(224, 224),
        # album.Normalize(mean=[0.7347, 0.4894, 0.6820, ], std=[0.1747, 0.2223, 0.1535, ]),
        ToTensorV2(),
    ])
    
    #===================================================================================================================
    # Test Inference
    #===================================================================================================================

    csv_path = './My_data/valid.csv'

    multi_load = True
    if multi_load:
        prefetch_factor = 4
    else:
        prefetch_factor = 2

    train_dataset = ListDataset(csv_path, 
                                batch_num=32, 
                                percentage=1,
                                transform=albumentations_valid, 
                                multi_load=True,
                                shuffle=True,
                                seed=0,
                                drop_last=False,
                                num_workers=4)
    print(len(train_dataset))
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=1, 
                              shuffle=False, 
                              num_workers=1, 
                              pin_memory=True, 
                              prefetch_factor=prefetch_factor, 
                              persistent_workers=False)
    print("Start loading")
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):   # enumerate(cycle(train_loader))  infinit cycle loader
        print(i)
        if i == 100:
            break
    elapse = time.time() - start_time
    print(f"bag_size: {images.size()}, {labels.size()}, elapse: {elapse:.4f}")


