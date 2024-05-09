import argparse
import csv
import glob
import os
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import confusion_matrix


def format_arg(args):
    msg = "\n".join("--%s=%s \\" % (k, str(v))
                    for k, v in dict(vars(args)).items())
    return "\n" + msg


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def pair(size):
    return tuple((size, size))


def get_fn_without_suffix(path):
    fn_suffix = os.path.basename(path)
    filename = fn_suffix.split('.')[0]
    return filename


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False):
    row = OrderedDict(epoch=epoch)
    row.update([('train_' + k, v) for k, v in train_metrics.items()])
    row.update([('valid_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=row.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(row)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class CutOut(object):
    """
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        value: erasing value. Default is 0. If a single int, it is used to
        erase all pixels. If a tuple of length 3, it is used to erase
        R, G, B channels respectively.
        If a str of 'random', erasing each pixel with random values.
    """

    def __init__(self,
                 p=0.5,
                 scale=(0.02, 0.3),
                 ratio=(0.2, 3.3),
                 value=0,
                 image_size=None):
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError(
                "Random erasing probability should be between 0 and 1")

        self.prob = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        i, j, h, w, value = self.get_params(img,
                                            scale=self.scale,
                                            ratio=self.ratio,
                                            value=self.value)
        img_draw = ImageDraw.Draw(img)
        img_draw.rectangle((i, j, h+i, w+j), fill=value, outline=None)
        return img

    def get_params(self, img, scale, ratio, value):

        img_w, img_h = img.size
        area = img_w * img_h
        if value == 'random':
            value = tuple(np.random.randint(210, 255, (3,)))
        log_ratio = np.log(ratio)
        for _ in range(10):
            erase_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(random.uniform(log_ratio[0], log_ratio[1]))

            h = int(round(np.sqrt(erase_area * aspect_ratio)))
            w = int(round(np.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            i = random.randint(0, img_h - h + 1)
            j = random.randint(0, img_w - w + 1)
            # print(f'i, j, h, w, {i}, {j}, {h}, {w}')
            return i, j, h, w, value
        return 0, 0, 0, 0, value


class PadToSquare(object):
    """
    Args:
        -max_size:(int), pad to max size
    """

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        w, h = img.size

        if w == h == self.target_size:
            return img

        max_size = max(w, h)
        min_size = min(w, h)

        ratio = self.target_size / max_size
        target_min_size = np.ceil(min_size * ratio).astype(int)

        if w > h:
            img_resize = img.resize(
                (self.target_size, target_min_size), resample=Image.Resampling.BICUBIC)
        else:
            img_resize = img.resize(
                (target_min_size, self.target_size), resample=Image.Resampling.BICUBIC)

        img_target = Image.new('RGB', size=(
            self.target_size, self.target_size))
        img_target.paste(img_resize)

        return img_target


class GetAlphaWithoutOne:
    def __init__(self, lower, upper):
        self.alpha = 1
        self.lower = lower
        self.upper = upper

    def __call__(self):
        while True:
            alpha = np.random.uniform(self.lower, self.upper, 1)[0]
            if alpha != self.alpha:
                return alpha


def random_mask_not_overlap(image_size, patch_size, ratio=0.5):
    image_height, image_width = (image_size, image_size)
    patch_height, patch_width = (patch_size, patch_size)

    assert image_height % patch_height == 0 and image_width % patch_width == 0, \
        'Image dimensions must be divisible by the patch size.'

    num_height_patches, num_width_patches = image_height // patch_height, image_width // patch_width
    num_patches = num_height_patches * num_width_patches

    indices = np.random.permutation(num_patches)

    selected = indices[:int(ratio * num_patches)]
    mask = np.ones((image_height, image_width), dtype=np.uint8)
    for i in range(num_height_patches):
        for j in range(num_width_patches):
            serial = i * num_height_patches + j
            condition = [serial in selected]
            if any(condition):
                height = i * patch_height
                width = j * patch_width
                mask[height:height+patch_height, width:width+patch_width] = 0
    return mask


class TimeConsume:
    def __init__(self, text=None):
        self.text = text

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()

        if self.text is None:
            print(f'time consume: {self.end - self.start:.4f}s')
        else:
            print(f'{self.text} time consume: {self.end - self.start:.4f}s')


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_loader_cv2(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img


def get_specified_files(folder_path,
                        suffixes=[".svs", ".tiff", ".tif"],
                        recursive=False):
    """
    Description:
        - Get all the suffixes files from folder path, or you could re-write it so that you could pass suffixes needed

    Parameters:
        - folder_path: str, The folder you want to get the files
        - suffixes   : list, list of all suffixes you want to get
        - recursive : bool, Which means if get from the folder of the folder_path or not. default is False

    Return:
        -  List of the files founded
    """

    files = []
    for suffix in suffixes:
        if recursive:
            path = os.path.join(folder_path, "**", "*" + suffix)
        else:
            path = os.path.join(folder_path, "*" + suffix)
        files.extend(glob.glob(path, recursive=recursive))
    return files


def create_save_path(save_path, second_dir=True):
    """
    Description:
        - Create save, weight, training_log folder for saving

    Parameters:
        - save_path: input, save path you want to save the weight, train log, etc

    Return:
        - list of images
    """
    path, folder = os.path.split(save_path)
    now = datetime.now()
    folder = "".join((folder, "_", now.strftime("%Y%m%d-%H%M%S")))
    save_path = os.path.join(path, folder)
    # #========================================================
    # files = os.listdir(path)
    # file_num = len(files)
    # if file_num > 0:
    #     save_path = os.path.join(path, f"{folder}_{file_num+1}")
    # else:
    #     save_path = os.path.join(path, f"{folder}_{0}")
    # #=============================================================
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    log_path, checkpoint_path = save_path, save_path
    if second_dir:
        checkpoint_path = os.path.join(save_path, 'weights')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        log_path = os.path.join(save_path, 'train_log')
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
    return log_path, checkpoint_path


def precision_recall_acc_fscore(y_true, y_pred, beta=1.0):
    """
    Description:
        - Calculate accuracy, precision, recall, f beta score, you could pass class label in string or integer

    Parameters:
        - y_true: input, ground truth labels
        - y_pred: input, prediction labels
        - beta: float or int, f beta score, default is 1.0

    Returns:
        - dict, contains accuracy, precision, recall, fscore
    """
    result = dict()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    unique = np.unique(y_true)
    for idx, item in enumerate(unique):
        TP = np.sum((y_pred == unique[idx]) & (y_pred == y_true))
        FP = np.sum((y_pred == unique[idx]) & (y_true != unique[idx]))
        FN = np.sum((y_true == unique[idx]) & (y_pred != unique[idx]))
        TN = np.sum((y_true != unique[idx]) & (y_pred != unique[idx]))

        P = FN + TP
        N = FP + TN

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        acc = (TP + TN) / (P + N)
        recall = TP / P
        if precision == 0 and recall == 0:
            f_score = 0.0
        else:
            f_score = (1 + beta**2) * (precision * recall) / \
                (beta**2 * precision + recall)

        result[item] = {
            "precision": precision,
            "accuracy": acc,
            "recall": recall,
            "fscore": f_score,
            "P": P
        }

    return result


def plot_confusion_matrix(y_true,
                          y_pred,
                          labels=None,
                          title=None,
                          figsize=None,
                          title_fontsize=None,
                          label_fontsize=None):
    """
    Description:
        - Plot confusion matrix

    Parameters:
        - y_true: array_like, true labels
        - y_pred: array_like, predict labels
        - labels: list_like, unique labels of the y_true or y_pred
        - title : str, title of the plot
        - figsize: tuple, figure size of the plot
        - title_fontsize: int, font size of the plot title
        - label_fontsize: int, font size of the plot x label & y label

    Return:
        - None        
    """
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')

    if title:
        plt.title('{}'.format(title), fontsize=title_fontsize)
    plt.xlabel('Predicted label', fontsize=label_fontsize)
    plt.ylabel('True label', fontsize=label_fontsize)

    plt.tight_layout()
    plt.show()


def resize_pad(img, width=224, height=224):
    # 保持纵横比例变成正方形
    if img.shape[1] > img.shape[0]:
        top = (img.shape[1] - img.shape[0]) // 2
        left = 0
        bottom = img.shape[1] - img.shape[0] - top
        right = 0
    elif img.shape[0] >= img.shape[1]:
        top = 0
        left = (img.shape[0] - img.shape[1]) // 2
        bottom = 0
        right = img.shape[0] - img.shape[1] - left

    img_with_border = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    img_resize = cv2.resize(img_with_border, (width, height), cv2.INTER_AREA)
    return img_resize


def get_biggest_roi(img, thres=15):
    # save_file_path
    # img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    if img.ndim == 3:
        img_gray = img[..., 0]
    else:
        img_gray = img.copy()
    # kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # img_5_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel_5)
    # img_5_close = cv2.morphologyEx(img_5_open, cv2.MORPH_CLOSE, kernel_5)

    _, image_bin = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel_5)
    contours, hierarchy = cv2.findContours(
        image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    # 得到最大的轮廓的点
    max_area_idx = np.argmax(np.array(area))
    cnt1 = contours[max_area_idx]
    x, y, w, h = cv2.boundingRect(cnt1)
    ratio = w/h

    # 取最大的轮廓进行扩增
    if 0.9 < ratio < 1.25:
        pad_l = 10
        pad_r = 20
    else:
        pad_l = 10
        pad_r = 10
    if (x - pad_l) >= 0:
        pnt_1_x = x - pad_l
    else:
        # pnt_1_x = x
        pnt_1_x = x  # 这样也可以兼顾到最左边

    if (y - pad_l) >= 0:
        pnt_1_y = y - pad_l
    else:
        pnt_1_y = y

    if x+w+pad_r < img_w:
        pnt_2_x = x + w + pad_r
    else:
        # pnt_2_x = x + w
        pnt_2_x = img_w  # 这个就可以取到最右边

    if y+h+pad_r < img_h:
        pnt_2_y = y + h + pad_l
    else:
        pnt_2_y = y + h

    # Test ROI
    # img_dst = cv2.rectangle(img, (pnt_1_x, pnt_1_y), (pnt_2_x, pnt_2_y), (0, 0, 255), 1)

    if img.ndim == 3:
        temp = img[pnt_1_y:pnt_2_y, pnt_1_x:pnt_2_x, :]
    else:
        temp = img[pnt_1_y:pnt_2_y, pnt_1_x:pnt_2_x, np.newaxis]

    # img_dst = resize_pad(temp)
    return temp


def random_brighten_img(img, **kargs):
    """
    Description:
        - random brighten image, math: dst = src1 * alpha + src2 * beta + gamma, alpha control brightness

    Parameters:
        - img: input uint8 [0, 255] RGB image
        - kargs: fixed: bool, if fixed alpha all the time, if True, alpha will have to passed
                 alpha: float, if alpha=1, then output should be same as input, 
                 if alpha < 1, will be darker, if alpha > 1, brighter
    """
    fixed = False
    if kargs:
        fixed = kargs['fixed']

    if fixed:
        alpha = kargs['alpha']
    else:
        contrast_prob = np.random.randint(0, 2)
        if contrast_prob == 1:
            alpha = np.random.uniform(0.7, 1.4, 1)[0]
        else:
            alpha = 1

    height, width, channels = img.shape
    beta = 1 - alpha
    blank = np.zeros([height, width, channels], img.dtype)
    dst = cv2.addWeighted(img, alpha, blank, beta, 1)
    return dst


def resize(img, size):
    if isinstance(size, tuple):
        size = size
    else:
        size = pair(size)
    return cv2.resize(img, size)


def random_flip(img, axis):
    """
    Description:
        - Random flip image, left right or up down

    Parameters:
        - img: input image
        - axis: int, if axis=0, is left_right flip, if axis=1 is up_down flip

    Return:
        - image flip or same depend on the prob
    """
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop:
        img = cv2.flip(img, axis)
    return img


def random_brightness(img, **kargs):
    """
    Description:
        - random image brightness, math: dst = src1 * alpha + src2 * beta + gamma, alpha control brightness

    Parameters:
        - img: input uint8 [0, 255] RGB image
        - kargs: fixed: bool, if fixed alpha all the time, if True, alpha will have to passed
                 alpha: float, if alpha=1, then output should be same as input, 
                 if alpha < 1, will be darker, if alpha > 1, brighter
    Returns:
        - return image transformed
    """
    fixed = False
    if kargs:
        fixed = kargs['fixed']

    if fixed:
        alpha = kargs['alpha']
    else:
        contrast_prob = np.random.randint(0, 2)
        if contrast_prob == 1:
            alpha = np.random.uniform(0.7, 1.4, 1)[0]
        else:
            alpha = 1

    height, width, channels = img.shape
    beta = 1 - alpha
    blank = np.zeros([height, width, channels], img.dtype)
    dst = cv2.addWeighted(img, alpha, blank, beta, 1)
    return dst


if __name__ == "__main__":
    # ############################### Test ROI ########################################
    # main()

    from PIL import Image
    val_folder = r"Data\some_tiles"
    val_images = get_specified_files(val_folder, ['.png'], recursive=False)

    img_ori = Image.open(val_images[0]).convert("RGB")
    get_alpha = GetAlphaWithoutOne(lower=0.85, upper=1.25)
    alpha = get_alpha()
    img = resize(np.array(img_ori), size=(480, 480))
    img = random_flip(img, 0)
    img = random_flip(img, 1)
    img_transform = random_brightness(img, fixed=True, alpha=alpha)

    show_list = ['img_ori', 'img_transform']

    plt.figure(figsize=(10, 5))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(eval(show_list[i]), 'gray')
        plt.axis("off")
        if show_list[i] == "img_transform":
            plt.title(f"Transform alpha = {alpha:.4f}")
        plt.subplots_adjust(hspace=0.01, wspace=0.001)
    # plt.tight_layout()
    plt.show()

    # ############################### Test TTA ########################################
    # import cv2
    # import albumentations as album
    # from predict import default_loader

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.load(r"./checkpoints\Efficientnet_b0_6\weights\best.pt", map_location=device)
    # print('Load Model Done!!!')
    # model.to(device)

    # image_path = r"D:\CVPJT\Project\Brain_PET_MR\Brain_MRI_PET\My_data\MRI\AD\1.png"
    # img0 = default_loader(image_path)
    # img_arr = np.array(img0)
    # albumentations_train = album.Compose([
    #     album.Resize(224, 224),
    #     album.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    #     # album.OneOf([album.MotionBlur(blur_limit=5), album.MedianBlur(blur_limit=3), album.GaussianBlur(blur_limit=3., sigma_limit=2.)], p=0.5),
    #     album.VerticalFlip(p=0.5),
    #     album.HorizontalFlip(p=0.5),
    #     album.ShiftScaleRotate(
    #         shift_limit=0.1,
    #         scale_limit=0.1,
    #         rotate_limit=10,
    #         interpolation=cv2.INTER_LINEAR,
    #         border_mode=cv2.BORDER_CONSTANT,
    #         p=1,
    #     ),
    #     album.Normalize(mean=[0.1125,0.1125,0.1125,], std=[0.2077,0.2077,0.2077,]),
    #     # ToTensorV2(),
    # ])
    # label_list = []
    # for i in range(100):
    #     label = test_time_augmentation(img_arr, model, device, albumentations_train, 5)
    #     label_list.append(label)

    # print(np.bincount(label_list))
