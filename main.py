import argparse
import os
import time

from tqdm import tqdm

from extract_patch import PathExtractor, get_dir_filename
from logger import get_localtime, logger
from utils import (format_arg, get_specified_files, str2bool,
                   supress_ctypes_warnings)

supress_ctypes_warnings()


parser = argparse.ArgumentParser()
parser.add_argument('--wsi-path', type=str, default=r'',
                    help='WSI folder path')
parser.add_argument('--wsi-suffix', nargs='+', default=['.tif', '.svs'],
                    help='Suffix of the WSI, you could get more if you need (default is [".tif"]).')
parser.add_argument('--recursive', type=str2bool, default=False,
                    help='Get WSI from wsi-path recursive or not (default is False).')
parser.add_argument('--save-path', type=str, default=r'',
                    help='Processed data save path.')
parser.add_argument('--ano-path', type=str, default=r'',
                    help='Where the annotation is.')
parser.add_argument('--ano-format', type=str, default='.xml',
                    help='Format of the annotation file, default is json, xml not implemented yet')
parser.add_argument('--xml-type', type=str, default='imagescope',
                    help='When ano_form is .xml, you should specify is imagescope or asap (defaul is imagescope)')
parser.add_argument('--csv-file', type=str, default=None,
                    help=f'Annotation csv file, annotation is not always at level 0, \
                           this file indicate which level the annotation is annotated.')
parser.add_argument('--extract-mode', type=str, default='annotation',
                    help='extract tile model, must be in [slide, annotation]')
parser.add_argument('--extract-region-name', nargs='+', default=('tumor without nerve',),
                    help='Name of the annotation region to extract, accept multiple names (default is tumor)')
parser.add_argument('--exclude-region-name', nargs='+', default=(),
                    help='Name of the annotation region to exclude, accept multiple names (default is empty)')
parser.add_argument('--mag', type=int, default=20,
                    help='which magnification to extract the tile')
parser.add_argument('--tile-size', type=int, default=512,
                    help='Extract tile size in square (default is 256 pixels)')
parser.add_argument('--overlap', type=int, default=0,
                    help='Tile overlap size, 0 mean non overlap')

# Random extraction instead of grid
parser.add_argument('--random_extract', type=str2bool, default=False,
                    help='Random extract tile from WSI instead of grid, (default is False)')
parser.add_argument('--sample_rate', type=float, default=1.0,
                    help='Sample rate for random extraction of tiles, (default is 1.0)')
parser.add_argument('--iou_thresh', type=float, default=0,
                    help='IOU threshod for tile overlap for random extraction.')

# Save parameters
parser.add_argument('--save-mask', type=str2bool, default=False,
                    help='save format for the extract tile, default is png')
parser.add_argument('--save-format', type=str, default='png',
                    help='save format for the extract tile, default is png')
parser.add_argument('--area_ratio', type=tuple, default=(0.2, 1.0),  # IoT
                    help='Area ratio interect with the region, (IoT default is (0.2, 1)).')
parser.add_argument('--object-thres', type=float, default=0.,  # ToT
                    help='object threshold to determine is background or foreground, (ToT=1-BoT, default is 0.1)')
parser.add_argument('--include-boundary', type=str2bool, default=False,
                    help='Include tile of the annotation boundary, True to include, (default is False).')
parser.add_argument('--boundary-min-thresh', type=int, default=1,
                    help='Minimum point(s) in(on) the region, (default is 3).')
parser.add_argument('--save-hdf5', type=str2bool, default=False,
                    help='if save the extracted tiles in hdf5, suffix is ".h5" ')
parser.add_argument('--num-workers', type=int, default=4,
                    help='multi thread workers for saving tile')

# Cell detect parameters
parser.add_argument('--cell_detect', type=str2bool, default=False,
                    help='If run the cell detect to detect the number of cell to determine to save tile (default False).')
parser.add_argument('--cell_detect_min_thresh', type=float, default=10,
                    help='Cell detection, minimum threshold for the Contour.')
parser.add_argument('--cell_detect_max_thresh', type=float, default=200,
                    help='Cell detection, maximum threshold for the Contour.')
parser.add_argument('--cell_detect_min_area', type=float, default=150,
                    help='Cell detection, minimum threshold for Area.')
parser.add_argument('--cell_detect_max_area', type=float, default=1500,
                    help='Cell detection, maximum threshold for Area..')
parser.add_argument('--cell_detect_min_circ', type=float, default=0.01,
                    help='Cell detection, minimum threshold for Circularity.')
parser.add_argument('--cell_detect_max_circ', type=float, default=None,
                    help='Cell detection, maximum threshold for Circularity..')
parser.add_argument('--cell_detect_min_conv', type=float, default=0.6,
                    help='Cell detection, minimum threshold for Convexity.')
parser.add_argument('--cell_detect_max_conv', type=float, default=None,
                    help='Cell detection, minimum threshold for Convexity..')
parser.add_argument('--cell_num_thresh', type=int, default=120,
                    help='Cell detection, number of cell threshold to save tile.')

# Random save parameters
parser.add_argument('--random-save', type=str2bool, default=False,
                    help='if random save or not')
parser.add_argument('--random-save-ratio', type=float, default=0.2,
                    help='random save ratio in uniform distribution')
parser.add_argument('--random-seed', type=int, default=42,
                    help='random seed')

# Normalize parameters
parser.add_argument('--normalize-mpp', type=str2bool, default=False,
                    help='normalize different mpp to same physical size or not, default is False')
parser.add_argument('--target-mpp', type=float, default=None,
                    help='normalize target mpp, typical is 0.25 in 40x')
parser.add_argument('--color-normalize', type=str2bool, default=False,
                    help='if do color normalization or not, not implement yet, Macenko, Vahadane, etc')

# test extract regions draw
parser.add_argument('--test-draw-extract-regions', type=str2bool, default=False,
                    help='test the draw extract region or not')


def main(args):
    # ========================== SYMH server ==========================
    kwargs = dict(save_path=args.save_path,
                  csv_file=args.csv_file,
                  extract_mode=args.extract_mode,
                  extract_region_name=args.extract_region_name,
                  exclude_region_name=args.exclude_region_name,
                  mag=args.mag,
                  tile_size=args.tile_size,
                  overlap=args.overlap,
                  save_mask=args.save_mask,
                  save_format=args.save_format,
                  object_thres=args.object_thres,
                  area_ratio=args.area_ratio,
                  cell_detect=args.cell_detect,
                  cell_detect_min_thresh=args.cell_detect_min_thresh,
                  cell_detect_max_thresh=args.cell_detect_max_thresh,
                  cell_detect_min_area=args.cell_detect_min_area,
                  cell_detect_max_area=args.cell_detect_max_area,
                  cell_detect_min_circ=args.cell_detect_min_circ,
                  cell_detect_max_circ=args.cell_detect_max_circ,
                  cell_detect_min_conv=args.cell_detect_min_conv,
                  cell_detect_max_conv=args.cell_detect_max_conv,
                  cell_num_thresh=args.cell_num_thresh,
                  num_workers=args.num_workers,
                  save_hdf5=args.save_hdf5,
                  random_extract=args.random_extract,
                  sample_rate=args.sample_rate,
                  iou_thresh=args.iou_thresh,
                  normalize_mpp=args.normalize_mpp,
                  target_mpp=args.target_mpp,
                  color_normalize=args.color_normalize,
                  random_save=args.random_save,
                  random_save_ratio=args.random_save_ratio,
                  random_seed=args.random_seed,
                  test_draw_extract_regions=args.test_draw_extract_regions)
    extractor = PathExtractor(**kwargs)
    wsis = get_specified_files(
        args.wsi_path, args.wsi_suffix, recursive=args.recursive)
    num_file = len(wsis)

    logger.info(
        f'Start extracting tiles from directory {args.wsi_path}, there are {num_file} slides')
    logger.info(f'Extractor kwargs: {format_arg(args)}')

    with tqdm(total=num_file) as pbar:
        for wsi in wsis[:]:
            dir, file_name = get_dir_filename(wsi)
            fn = file_name.split('.')[0]
            pbar.set_description(f'Extracting tiles: {fn}')
            start_time = time.time()
            if args.extract_mode == 'annotation':
                ano_file = os.path.join(args.ano_path, fn + args.ano_format)
                if not os.path.exists(ano_file):
                    logger.info(f'{ano_file} not exists')
                    continue
            else:
                ano_file = None
            try:
                extractor.extract(wsi, ano_file, xml_type=args.xml_type)
            except Exception as e:
                logger.exception(e)
            end_time = time.time()
            logger.info(f'{wsi} Done!!! time: {end_time - start_time:.4f}')

            pbar.update(1)


if __name__ == '__main__':
    logger.set_level(2, 0)
    time_message = get_localtime()
    logger.set_formatter(None)
    logger.info(time_message.center(120, '*'))
    logger.reset_default_formatter()
    args = parser.parse_args()
    main(args)
