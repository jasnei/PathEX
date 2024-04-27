import itertools
import json
import os
import random
import time
from queue import Queue
from threading import Thread
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from shapely import geometry

from logger import get_localtime, logger
from utils import get_dir_filename, supress_ctypes_warnings

supress_ctypes_warnings()


def open_slide(file_path):
    try:
        slide = openslide.OpenSlide(file_path)
    except openslide.OpenSlideError:
        logger.exception("OpenSlideError: {}".format(file_path))
        slide = None
    return slide


def show_slide(slide, size=None):
    if size is None:
        size = (512, 512)
    plt.imshow(slide.get_thumbnail(size))
    plt.show()


def get_slide_properties(slide) -> Dict:
    """some slide properties might not be saved rightly, we need to generate the 
    properties information from slide detail. 
    We need level-count, and level width, height and downsample.
    """
    try:
        properties = {k: v for k, v in slide.properties.items()}
    except:
        # Calculate the width and height if the properties not specified
        properties = {}
        level_count = len(slide.level_dimensions)
        properties['openslide.level-count'] = str(level_count)

        level_0_width, level_0_height = slide.dimensions
        for i in range(level_count):
            width, height = slide.level_dimensions[i]
            downsample = level_0_width / width
            # print(i, width, height, downsample)
            properties[f'openslide.level[{i}].width'] = str(width)
            properties[f'openslide.level[{i}].height'] = str(height)
            properties[f'openslide.level[{i}].downsample'] = str(downsample)

    # Some older versions
    try:
        mpp_x = float(properties['openslide.mpp-x'])
    except KeyError:
        mpp_x = 0

    try:
        mpp_y = float(properties['openslide.mpp-y'])
    except KeyError:
        mpp_y = 0

    try:
        mag = int(properties['openslide.objective-power'])
    except KeyError:
        try:
            mag = int(properties['aperio.AppMag'])
        except KeyError:
            mag = 0

    # Try to fix magnifications
    if mag == 0 and mpp_x != 0:
        if 0.22 < mpp_x <= 0.26:
            mag = 40
        elif 0.44 < mpp_x <= 0.52:
            mag = 20
        elif 0.88 < mpp_x <= 1.04:
            mag = 10
    metrics = dict(mpp_x=mpp_x, mpp_y=mpp_y, mag=mag)
    properties.update(metrics)
    return properties


def get_file_name_wo_suffix(file_path):
    return os.path.basename(file_path).split('.')[0]


def generate_save_dir(file_path, save_path, random_extract=False, level=None, tile_size=None):
    slide_name = get_file_name_wo_suffix(file_path)
    if random_extract:
        save_dir = os.path.join(save_path, 'random_' + slide_name)
    else:
        save_dir = os.path.join(save_path, slide_name)
    if level is not None:
        save_dir = save_dir + f'_level_{level}'
    if tile_size is not None:
        save_dir = save_dir + f'_{tile_size}'
    save_dir = os.path.join(save_dir, 'tiles')
    return save_dir, slide_name


def mag_to_level(mag, mag_target, mags=(40, 20, 10, 5, 3, 2, 1)):
    if not mag_target in mags:
        logger.warning(f'mag must be one of {mags}, but got {mag_target}')
        return None
    if mag_target > mag:
        logger.warning(
            f'slide max magnification is {mag}, but target magnification is {mag_target}')
        return None

    if mag == 40:
        level = mags.index(mag_target)
    elif mag == 20:
        mags = mags[1:]
        level = mags.index(mag_target)
    elif mag == 10:
        mags = mags[2:]
        level = mags.index(mag_target)
    elif mag == 5:
        mags = mags[2:]
        level = mags.index(mag_target)
    else:
        level = None
        # raise NotImplementedError
    return level


def parse_json_annotation(json_path):
    logger.info('parsing json annotation: {}'.format(json_path))
    with open(json_path, mode='r', encoding='utf-8') as f:
        json_data = json.load(f)
    try:
        # features
        features = json_data['features']
    except TypeError:
        features = json_data
    except KeyError:
        features = json_data

    # geometries and properties
    geometries = []
    properties = []
    for feature in features:
        geometries.append(feature['geometry'])
        properties.append(feature['properties'])

    # classification_names
    classification_names = []
    for property in properties:
        try:
            classification_names.append(
                property['classification']['name'].lower())
        except KeyError:
            classification_names.append('none')

    # coordinates, annotation could be Multipolygon or Polygon
    geometry_types = []
    coordinates = []
    for i, geometry in enumerate(geometries):
        geometry_type = geometry['type']
        if geometry_type.lower() == 'polygon':
            coordinate = geometry['coordinates']
            if len(coordinate) > 1:
                coordinate = np.array(coordinate[0], dtype=np.int32)
            else:
                coordinate = np.array(coordinate, dtype=np.int32)
        elif geometry_type.lower() == 'multipolygon':
            coordinate = np.array(
                geometry['coordinates'][0][0], dtype=np.int32)
        else:
            coordinate = np.array(geometry['coordinates'][0], dtype=np.int32)
        coordinates.append(coordinate)
        geometry_types.append(geometry_type)

    return coordinates, classification_names


def parse_xml_annotation(ano_file, tag='Annotation', attrib_name='PartOfGroup',
                         tumor_group=tuple(('_0', '_1', 'tumor')), normal_group=('_2',), xml_type='imagescope'):
    # process
    try:
        tree = ET.parse(ano_file)
    except:
        logger.exception(f'Could not parse {ano_file}')
        raise FileNotFoundError

    coords = []
    names = []
    root = tree.getroot()
    for ano in root.iter(tag=tag):
        ano_attrib = ano.attrib  # dict, sometimes will be empty

        # Process annotation name for different xml types
        if xml_type == 'imagescope':
            ano_name = ano_attrib[attrib_name].lower()
            group_name = ano_name
        else:
            ano_name = 'normal'
            if ano_attrib:
                group_name = ano_attrib[attrib_name].lower()
                if group_name in tumor_group:
                    ano_name = 'tumor'
                elif group_name in normal_group:
                    ano_name = 'normal'

        for regions in ano:
            regions_tag = regions.tag.lower()
            if regions_tag in ('coordinates', 'plots'):
                # ==========for ASAP========================================
                # ASAP annotation only two layers
                temp = []
                for coordinate in regions:
                    x = float(coordinate.attrib['X'])
                    y = float(coordinate.attrib['Y'])
                    temp.append([x, y])

                if temp:
                    coords.append(
                        np.array(temp, dtype=np.int32).reshape(1, -1, 2))
                    names.append((group_name, ano_name))
                # =========================================================
            elif regions_tag not in ('regions'):
                continue
            else:
                # ================= Below is for Imagescope ================
                # Imagescope annotation four layers
                for region in regions:
                    region_tag = region.tag.lower()
                    if region_tag not in ('region', 'plots'):
                        continue
                    region_name = region.attrib["Text"]
                    for vertices in region:
                        if vertices.tag.lower() not in ('vertices', 'coordinates'):
                            continue
                        temp = []
                        for vertiex in vertices:
                            # print(vertiex.attrib['X'])
                            x = float(vertiex.attrib['X'])
                            y = float(vertiex.attrib['Y'])
                            temp.append([x, y])

                        if temp:
                            coords.append(
                                np.array(temp, dtype=np.int32).reshape(1, -1, 2))
                            names.append((ano_name, region_name))
                # ==============================================================
    return coords, names


def save(image, filename):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    state = cv2.imwrite(filename, image)
    # print('{}, shap {}, saved: {}'.format(filename, image.shape, state))
    return None


def get_white_mask(img, lower=[0, 0, 160], upper=[180, 30, 255], inv=True):
    # # white background lower and upper, lower could be smaller
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if not isinstance(lower, np.ndarray):
        lower = np.array(lower)

    if not isinstance(upper, np.ndarray):
        upper = np.array(upper)

    mask = cv2.inRange(img_hsv, lower, upper)
    if inv:
        mask = ~mask

    return mask


def get_mask_area(mask, contour=True):
    """
    Description:
        - Get mask area and and total area

    Args:
        - mask: (numpy.ndarray), input binary mask
        - contour: (bool), if get contour area or nonezero
    """
    w, h = mask.shape[:2]
    if contour:
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt, False)
            areas.append(area)
        area = np.sum(areas)
        area_total = (w - 1) * (h - 1)
    else:
        area = cv2.countNonZero(mask)
        area_total = w * h

    return area, area_total


def get_white_region_ratio(img, blur_size=(5, 5), mask_inv=False, contour=True, eps=1e-7):
    img = cv2.GaussianBlur(img, blur_size, 0, 1)
    mask = get_white_mask(img, inv=mask_inv)

    # dilate the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
    mask = ~cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    area, area_total = get_mask_area(mask, contour=contour)
    ratio = area / (area_total + eps)
    return ratio


def is_background(img, object_thres=0.1):
    """
    input image must be in RGB order
    object_thres default is 0.1, means 10% is cover by object, and  the immage 90% is white. you could use 
    this object_thres to control how much the region is to determin whether tile is background or foreground object
    """
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    ratio = get_white_region_ratio(img)
    state = False
    if ratio <= object_thres:
        state = True
    return state


def compute_intersection_area(multipolygons, rectangle):
    """
    return intersection area of polygon and rectangle
    """
    # print(f'type of multipolygons: {type(multipolygons)}, {type(rectangle)}')

    # with np.errstate(invalid='ignore'): # This will supress the intersects RuntimeWarning, but slow down
    state = multipolygons.intersects(rectangle)

    if state:
        intersection = multipolygons.intersection(rectangle)
        return intersection.area, intersection
    else:
        return 0, None

    # intersection = multipolygons.intersection(rectangle)
    # if intersection.is_empty:
    #     return 0, intersection
    # return intersection.area, intersection


def get_rectangle_fourpoints(loc):
    """generate rectangle four points, return in four points anti-clockwise"""
    top_left = loc[0]
    bottom_right = loc[1]
    top_right = (bottom_right[0], int(top_left[1]))
    bottom_left = (int(top_left[0]), bottom_right[1])
    return top_left, bottom_left, bottom_right, top_right


def tile_is_in_regions(tile_polygon, multipolygons, area_ratio=(0.20, 1.0)):
    """
    tile is in regions, or interect with regions

    Args:
    - top_left: (int, int), tuple of (int, int)
    - w: (int), width of the tile
    - h: (int), height of the tile
    - regions: (shapely.geometry.multipolygon.MultiPolygon), 
    - area_ratio: (float, flaot), tuple of (float, float), is range of area ratio from 0.0 to 1.0, (0, 1.0]
    """
    tile_area = tile_polygon.area

    # Regions is not only one, compare with all regions, instead of just one region
    area, intersection = compute_intersection_area(multipolygons, tile_polygon)
    ratio = area / tile_area
    if area_ratio[0] < ratio and ratio <= area_ratio[1]:
        return True, intersection
    return False, None


def points_to_contour(four_points):
    # top_left, top_right, bottom_left, bottom_right = generate_rectangle_fourpoints(top_left, w, h)
    contour = np.array([four_points]).reshape(-1, 1, 2)
    return contour


def create_detector(min_thresh=10,
                    max_thresh=200,
                    min_area=150,
                    max_area=1500,
                    min_circ=0.01,
                    max_circ=None,
                    min_conv=0.6,
                    max_conv=None):

    detector = cv2.SimpleBlobDetector()
    # Setup parameters
    params = cv2.SimpleBlobDetector_Params()

    # Color
    # params.blobColor = 0

    # Thresholds
    if min_thresh is not None:
        params.minThreshold = min_thresh
    if max_thresh is not None:
        params.maxThreshold = max_thresh

    # Filter by area
    params.filterByArea = True
    if min_area is not None:
        params.minArea = min_area
    if max_area is not None:
        params.maxArea = max_area

    # Filter by circularity
    params.filterByCircularity = True
    if min_circ is not None:
        params.minCircularity = min_circ
    if max_circ is not None:
        params.minCircularity = max_circ

    # Filter by convexity
    params.filterByConvexity = True
    if min_conv is not None:
        params.minConvexity = min_conv
    if max_conv is not None:
        params.minConvexity = max_conv

    # Filter by inertia
    params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    # params.maxInertiaRatio = 10

    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def simple_cell_detect(img, detector):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoints = detector.detect(img)
    return len(keypoints)


def is_page1_thumbnail(file_path):
    """Check page 1 is thumbnail image or not. Some slide thumbnails is page 1, but some slide thumbnails
        is last page (Phlips, or the sequence of the page maybe altered by someone).
    """
    tif = tifffile.TiffFile(file_path)
    pages = tif.pages
    page0 = pages[0]
    page1 = pages[1]
    page_ratio = page0.imagelength/page1.imagelength  # if page_ratio < 2.5
    if page_ratio < 2.5:
        page1_is_thumbnail = False
    else:
        page1_is_thumbnail = True
    return page1_is_thumbnail


def polygon_to_multipolygon(coords) -> list:
    polys = []
    for coord in coords:
        if (coord is not None and coord.shape[1] > 2):
            poly = geometry.Polygon(coord.squeeze(0)).buffer(0.001)
            if poly.geom_type.lower() == 'multipolygon':
                for pl in poly.geoms:
                    polys.append(pl)
            else:
                polys.append(poly)
    return polys


def fill_contour_to_image(image, contours, color):
    mask = np.zeros(image.shape, np.uint8)
    for cnt in contours:
        mask = cv2.fillPoly(mask, [cnt], color=color)
    image = cv2.addWeighted(image, 1.0, mask, 0.2, 0.0)
    return image


def draw_extract_regions(file_path, slide_properties, file_name, num_workers, level, locations,
                         extract_coordinates, exclude_coordinates, extract_mode, object_thres,
                         area_ratio, random_save, random_save_ratio, cell_detector=None,
                         cell_num_thresh=0, thickness=2, color=(0, 255, 0), filled=False, scale_factor=64):
    """
    Description:
        - draw extract tile regions in multi threads, that would be save lots time rather than single thread, you could set num_workers=1 enable single thread

    Arguments:
        - file_path: (str), input slide file path
        - file_name: (str), save file name for the processed png image
        - num_workers: (int), how many workers process
        - level: (int), which level of the slide to process
        - location: (list), location deepzoom generator
        - deep_gen: (generator), deepzoom generator
        - extractor_coordinates: (list), list of coordinate to extract
        - extract_mode: (str), model to extract, one of (slide, annotation)
        - object_thres: (float), determine how much object is to be considered as backgroud, default is 0.1,
                        means if less then 10% object, then will be consider as background.
        - thickness: (int), thickness of the contour line
        - color: (tuple), (r, g, b) color of the contour line, default is (0, 255, 0) which means green
        - scale_factor: (int), scale factor for the output image to show from the level you extract the tile.

    Return:
        - None        
    """
    # deep_level = deep_gen.level_count - 1
    # extract regions should be also scale to target size
    tif = tifffile.TiffFile(file_path)
    pages = tif.pages
    page1_is_thumbnail = is_page1_thumbnail(file_path)

    if level == 0:
        source_image = pages[0].asarray()
    else:
        if page1_is_thumbnail:
            key = level + 1
        else:
            key = level

        # tifffile pages is fast and simple, but sometimes have index error, and it's not caused by the "key" we pass
        # openslide could fix this bug, but openslide needs more than twice memory which will cause memmory overflow
        # One could fix this is try to fix the tifffile bug in line 8161
        try:
            source_image = pages[key].asarray()
        except:
            print('reading source image using openslide')
            slide = openslide.OpenSlide(file_path)
            source_image = slide.read_region(
                (0, 0), key, (slide.level_dimensions[key])).convert('RGB')
            source_image = np.asarray(source_image, dtype=np.uint8)
    down_sample = int(
        float(slide_properties[f'openslide.level[{level}].downsample']))

    logger.info(
        f'source_image shape: {source_image.shape}, level: {level}, down_sample: {down_sample}, page1_is_thumbnail: {page1_is_thumbnail}')

    # # extract and exclude coordinates
    # extract_coordinates = [np.array(region // down_sample).astype(np.int32) for region in extract_coordinates]
    # exclude_coordinates = [np.array(region // down_sample).astype(np.int32) for region in exclude_coordinates]

    # Turn coordinates to polygon, add buffer avoid errors
    extract_coordinates_poly = geometry.MultiPolygon(
        polygon_to_multipolygon(extract_coordinates)).buffer(0.01)
    exclude_coordinates_poly = geometry.MultiPolygon(
        polygon_to_multipolygon(exclude_coordinates)).buffer(0.01)

    # multi threads, if num_workers=1, would be single thread
    tile_que = Queue()
    save_que = Queue()
    contours = tiling_for_contour(tile_que, save_que, num_workers, locations,
                                  extract_coordinates_poly, exclude_coordinates_poly,
                                  extract_mode, object_thres, area_ratio, random_save, random_save_ratio,
                                  cell_detector, cell_num_thresh)

    ratio = 1 / scale_factor
    image = cv2.resize(source_image, None, fx=ratio, fy=ratio)
    contours = [(item * ratio).astype(np.int32) for item in contours]
    if extract_mode == 'annotation':
        extract_coordinates = [(region * ratio).astype(np.int32)
                               for region in extract_coordinates]
        image = cv2.drawContours(
            image, extract_coordinates, -1, (0, 100, 0), thickness+1)

    # Show contours lines
    image = cv2.drawContours(image, contours, -1, color, thickness)

    # Fill contours
    if filled:
        image = fill_contour_to_image(image, contours, color)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    state = cv2.imwrite(file_name, image)
    logger.info(
        f"num_tiles: {len(contours)}, save state: {state}, file name: {file_name}")
    return None


def get_tile_location(slide, slide_properties, level, tile_size,
                      overlap, save_dir, slide_name, save_format, include_bounds_size):
    """Get the tile location according to level width and height, non-overlap or overlap
    both work."""

    # slide_height_l0 = int(slide_properties[f'openslide.level[0].height'])
    # slide_width_l0 = int(slide_properties[f'openslide.level[0].width'])
    slide_height = int(slide_properties[f'openslide.level[{level}].height'])
    slide_width = int(slide_properties[f'openslide.level[{level}].width'])
    downsample = int(
        float(slide_properties[f'openslide.level[{level}].downsample']))
    num_tiles_height = int(np.ceil(slide_height / (tile_size - overlap)))
    num_tiles_width = int(np.ceil(slide_width / (tile_size - overlap)))

    mask_dir = os.path.dirname(save_dir)
    tile_locations = []
    for y, x in itertools.product(range(num_tiles_height), range(num_tiles_width)):
        x_l = x * (tile_size - overlap)
        y_l = y * (tile_size - overlap)
        x_delta_l = slide_width - x_l
        y_delta_l = slide_height - y_l

        x_l0 = x_l * downsample
        y_l0 = y_l * downsample

        if (x_delta_l < tile_size) or (y_delta_l < tile_size):
            if x_delta_l >= include_bounds_size[0] and y_delta_l >= include_bounds_size[1]:
                tile_width_l = x_delta_l
                tile_height_l = y_delta_l
            else:
                continue
        else:
            tile_width_l = tile_size
            tile_height_l = tile_size

        loc_l = ((x_l, y_l), (x_l + tile_width_l, y_l + tile_height_l))
        loc_l0 = ((x_l0, y_l0), level, (tile_width_l, tile_height_l))
        loc_info = ((y_l0, x_l0), level, (tile_width_l, tile_height_l))

        tile_name = ''.join((slide_name, '_', str(loc_info), save_format))
        filename = os.path.join(save_dir, tile_name)
        maskname = os.path.join(mask_dir, 'masks', tile_name)
        prob = random.uniform(0, 1)

        tile_info = dict(slide=slide, loc=loc_l0, loc_l=loc_l,
                         filename=filename, maskname=maskname, prob=prob)
        tile_locations.append(tile_info)

    return tile_locations, num_tiles_width, num_tiles_height


def random_generate_tile_locations(slide, slide_properties, level, tile_size, save_dir, extract_mode,
                                   slide_name, save_format, extract_coordinates, sample_rate=2, iou_thresh=0.5):
    if extract_mode == 'slide':
        locations, num_tiles_width, num_tiles_height = generate_slide_tile_locations(slide, slide_name, slide_properties, level, tile_size,
                                                                                     save_dir, save_format, sample_rate, iou_thresh)
    else:
        locations, num_tiles_width, num_tiles_height = generate_region_tile_locations(slide, slide_name, slide_properties, level, tile_size,
                                                                                      save_dir, save_format, extract_coordinates, sample_rate, iou_thresh)
    return locations, num_tiles_width, num_tiles_height


def generate_slide_tile_locations(slide, slide_name, slide_properties, level, tile_size, save_dir, save_format, sample_rate, iou_thresh):
    slide_height = int(slide_properties[f'openslide.level[{level}].height'])
    slide_width = int(slide_properties[f'openslide.level[{level}].width'])
    downsample = int(
        float(slide_properties[f'openslide.level[{level}].downsample']))

    num_tiles_height = int(np.ceil(slide_height / tile_size))
    num_tiles_width = int(np.ceil(slide_width / tile_size))

    counts = int((slide_width * slide_height) //
                 (tile_size*tile_size) * sample_rate)
    tls = np.random.randint(
        (0, 0), (slide_width-tile_size, slide_height-tile_size), size=(counts, 2))

    # Threre are about 10,000 patch generated, so filtering centroids will be very slow
    # might be considered to generate centroids from tissue instead of whole slide
    # you could uncomment the following line to enable filtering function, it double the time consuming
    tls = iou_filter_locations(tls, 0, 0, tile_size, iou_thresh)

    tile_locations = []
    for x_l, y_l in tls:
        br = (x_l + tile_size, y_l + tile_size)

        x_l0 = x_l * downsample
        y_l0 = y_l * downsample

        loc_l = ((x_l, y_l), br)
        loc_l0 = ((x_l0, y_l0), level, (tile_size, tile_size))
        loc_info = ((y_l0, x_l0), level, (tile_size, tile_size))

        tile_name = ''.join((slide_name, '_', str(loc_info), save_format))
        filename = os.path.join(save_dir, tile_name)
        prob = random.uniform(0, 1)

        tile_info = dict(slide=slide, loc=loc_l0, loc_l=loc_l,
                         filename=filename, prob=prob)
        tile_locations.append(tile_info)

    return tile_locations, num_tiles_width, num_tiles_height


def generate_region_tile_locations(slide, slide_name, slide_properties, level, tile_size, save_dir,
                                   save_format, extract_coordinates, sample_rate=2, iou_thresh=0.8):
    mask_dir = os.path.dirname(save_dir)
    slide_height = int(slide_properties[f'openslide.level[{level}].height'])
    slide_width = int(slide_properties[f'openslide.level[{level}].width'])
    downsample = int(
        float(slide_properties[f'openslide.level[{level}].downsample']))

    num_tiles_height = int(np.ceil(slide_height / tile_size))
    num_tiles_width = int(np.ceil(slide_width / tile_size))

    tile_locations = []
    area_t = tile_size * tile_size
    for coord in extract_coordinates:
        try:
            coord = coord.squeeze(0)
        except:
            pass
        area = cv2.contourArea(coord)
        area_ratio = area / area_t
        if area_ratio >= 10:
            n_sample = area_ratio * 4
        elif area_ratio > 3:
            n_sample = area_ratio * 2
        else:
            n_sample = area_ratio + 1
        n_sample = int(n_sample * sample_rate)
        wh_min = np.min(coord, axis=0) - tile_size // 2
        wh_max = np.max(coord, axis=0) + 20

        tls = np.random.randint(low=wh_min, high=wh_max,
                                size=(n_sample, 2)).tolist()
        tls = iou_filter_locations(tls, 0, 0, tile_size, iou_thresh)
        # print(n_sample, len(tls))
        mask_dir = os.path.dirname(save_dir)
        for x_l, y_l in tls:
            br = (x_l + tile_size, y_l + tile_size)

            x_l0 = x_l * downsample
            y_l0 = y_l * downsample

            loc_l = ((x_l, y_l), br)
            loc_l0 = ((x_l0, y_l0), level, (tile_size, tile_size))
            loc_info = ((y_l0, x_l0), level, (tile_size, tile_size))
            # print(loc_l, loc_l0)
            tile_name = ''.join((slide_name, '_', str(loc_info), save_format))
            filename = os.path.join(save_dir, tile_name)
            maskname = os.path.join(mask_dir, 'masks', tile_name)
            prob = random.uniform(0, 1)

            tile_info = dict(slide=slide, loc=loc_l0, loc_l=loc_l,
                             filename=filename, maskname=maskname, prob=prob)
            tile_locations.append(tile_info)

    return tile_locations, num_tiles_width, num_tiles_height


def generate_bbox_wh(top_left, w, h):
    """[y, x, y+h, x+w]"""
    return np.array([top_left[1], top_left[0], top_left[1] + h, top_left[0] + w]).reshape(-1, 4)


def iou_xyxy(box_left, box_right):
    # left_tope [xmin, ymin]
    lt = np.maximum(box_left[:, None, :2], box_right[:, :2])
    # right_bottom [xmax, ymax]
    rb = np.minimum(box_left[:, None, 2:], box_right[:, 2:])
    # inter_area [inter_w, inter_h]
    wh = np.maximum(rb - lt + 1, 0)
    intersection = wh[..., 0] * wh[..., 1]                    # shape: (n, m)
    s1 = (box_left[:, 2] - box_left[:, 0] + 1) * \
        (box_left[:, 3] - box_left[:, 1] + 1)
    s2 = (box_right[:, 2] - box_right[:, 0] + 1) * \
        (box_right[:, 3] - box_right[:, 1] + 1)
    union = s1 + s2 - intersection
    iou = intersection / union
    return iou


def iou_filter_locations(tls, delta_x, delta_y, tile_size, iou_thresh=0.5):
    """
    Filter centers according to the IOU of each generated bounding box 

    Args:
        - centroids: (list) list of centroid generated [[int, int], ..., [int, int]]
        - delta_x: (int) half of width of the tile size
        - delta_y: (int) half of height of the tile size
        - tile_size: (int) tile size
        - iou_thresh: (float) determine how much of the overlap between the tiles

    Returns:
        - List of centroids
    """
    if isinstance(tls, np.ndarray):
        tls = tls.tolist()

    bboxes = []
    for x, y in (tls):
        bbox = generate_bbox_wh((x, y), tile_size, tile_size)
        bboxes.append(bbox)

    # process iou to determine how much overlap
    tls_left = []
    for i in range(len(bboxes)):
        # always pop first bbox to compare with balances
        bbox_i = bboxes.pop(0)
        tl_i = tls.pop(0)
        tls_left.append(tl_i)
        # if list is empty, break the loop
        if len(bboxes) <= 0:
            break
        # compare bboxes IOU
        bbox_res = np.concatenate(bboxes)
        ious = iou_xyxy(bbox_i, bbox_res)
        condition = np.array(ious).flatten() >= iou_thresh
        if any(condition):
            indices, = np.nonzero(condition)
            # reverse indices so that could remove larger indices
            for idx in indices[::-1]:
                bboxes.pop(idx.item())
                tls.pop(idx.item())
        if len(bboxes) <= 0:
            break
    return tls_left


def tile_contour(tile_que, save_que, random_save, random_save_ratio,
                 extract_coordinates, exclude_coordinates,
                 extract_mode, object_thres, area_ratio,
                 cell_detector=None, cell_num_thresh=0):
    while True:
        data = tile_que.get()
        if data is None:
            break

        slide = data['slide']
        args = data['loc']
        level_size = args[-1]
        loc_l = data['loc_l']
        prob = data['prob']

        if extract_mode == 'annotation':
            # same as tile_polygon.exterior.coords[:-1]
            four_points = get_rectangle_fourpoints(loc_l)
            minx, miny, maxx, maxy = loc_l[0][0], loc_l[0][1], loc_l[1][0], loc_l[1][1]
            tile_polygon = geometry.box(minx, miny, maxx, maxy)

            P, _ = tile_is_in_regions(
                tile_polygon, extract_coordinates, area_ratio)
            Q, _ = tile_is_in_regions(
                tile_polygon, exclude_coordinates, area_ratio)
            if P and not Q:
                tile = get_tile(slide, level_size, args)

                if is_background(tile, object_thres):
                    continue

                if random_save:
                    if prob > random_save_ratio:
                        continue
                    if cell_detector is not None:
                        cell_num = simple_cell_detect(tile, cell_detector)
                        if cell_num < cell_num_thresh:
                            continue
                else:
                    if cell_detector is not None:
                        cell_num = simple_cell_detect(tile, cell_detector)
                        if cell_num < cell_num_thresh:
                            continue
                save_que.put(points_to_contour(four_points))
        elif extract_mode == 'slide':
            if random_save:
                if prob > random_save_ratio:
                    continue
                four_points = get_rectangle_fourpoints(loc_l)
                tile = get_tile(slide, level_size, args)
                if is_background(tile, object_thres):
                    continue
                if cell_detector is not None:
                    cell_num = simple_cell_detect(tile, cell_detector)
                    if cell_num < cell_num_thresh:
                        continue
                save_que.put(points_to_contour(four_points))
            else:
                four_points = get_rectangle_fourpoints(loc_l)
                tile = get_tile(slide, level_size, args)
                if is_background(tile, object_thres):
                    continue
                if cell_detector is not None:
                    cell_num = simple_cell_detect(tile, cell_detector)
                    if cell_num < cell_num_thresh:
                        continue
                save_que.put(points_to_contour(four_points))
    return None


def tiling_for_contour(tile_que, save_que, num_workers, locations,
                       extract_coordinates, exclude_coordinates,
                       extract_mode, object_thres, area_ratio, random_save,
                       random_save_ratio, cell_detector=None, cell_num_thresh=0):
    if extract_mode == 'annotation':
        process_workers = num_workers
    else:
        process_workers = num_workers * 4

    tile_threads = []
    for _ in range(1):
        tile_thread = Thread(target=tile_to_que, args=(locations, tile_que))
        tile_thread.start()
        tile_threads.append(tile_thread)

    process_threads = []
    for _ in range(process_workers):
        process_thread = Thread(target=tile_contour, args=(tile_que, save_que, random_save, random_save_ratio,
                                                           extract_coordinates, exclude_coordinates,
                                                           extract_mode, object_thres, area_ratio,
                                                           cell_detector, cell_num_thresh))
        process_thread.start()
        process_threads.append(process_thread)

    for tile_thread in tile_threads:
        tile_thread.join()

    for _ in process_threads:
        tile_que.put(None)

    for process_thread in process_threads:
        process_thread.join()

    contours = list(save_que.queue)
    return contours


def tile_to_que(locations, tile_que):
    for location in locations:
        tile_que.put(location)
    return None


def get_tile_mask(tile_polygon, intersection, tile_size, outline=255, fill=255):
    tile_mask = Image.new('L', (tile_size, tile_size))
    bl = tile_polygon.exterior.coords[3]
    if intersection and intersection.geom_type.lower() == 'multipolygon':
        for poly in intersection.geoms:
            poly_mask = [(pt[0] - bl[0], pt[1] - bl[1])
                         for pt in poly.exterior.coords]
            ImageDraw.Draw(tile_mask).polygon(
                poly_mask, outline=outline, fill=fill)
    else:
        mask_polygon = [(pt[0] - bl[0], pt[1] - bl[1])
                        for pt in intersection.exterior.coords]
        ImageDraw.Draw(tile_mask).polygon(
            mask_polygon, outline=outline, fill=fill)
    return tile_mask


def get_tile(slide, size, args):
    tile = slide.read_region(*args)
    if tile.size != size:
        tile.thumbnail(size, getattr(Image, 'Resampling', Image).LANCZOS)
    return tile


def process_tile(tile_que, save_que, tile_size, extract_coordinates, exclude_coordinates,
                 random_save, random_save_ratio, extract_mode, object_thres, area_ratio,
                 cell_detector=None, cell_num_thresh=0):
    while True:
        data = tile_que.get()
        if data is None:
            break

        slide = data['slide']
        args = data['loc']
        level_size = args[-1]
        loc_l = data['loc_l']

        tile_name = data['filename']
        prob = data['prob']

        if extract_mode == 'annotation':
            minx, miny, maxx, maxy = loc_l[0][0], loc_l[0][1], loc_l[1][0], loc_l[1][1]
            tile_polygon = geometry.box(minx, miny, maxx, maxy)

            P, intersection_extract = tile_is_in_regions(
                tile_polygon, extract_coordinates, area_ratio)
            Q, intersection_exclude = tile_is_in_regions(
                tile_polygon, exclude_coordinates, area_ratio)
            if P and not Q:
                tile = get_tile(slide, level_size, args)
                # print(type(intersection_extract))
                if is_background(tile, object_thres):
                    continue
                # TODO: here could do some transformation for the tile: color normalization

                if random_save:
                    if prob > random_save_ratio:
                        continue
                    if cell_detector is not None:
                        cell_num = simple_cell_detect(tile, cell_detector)
                        if cell_num < cell_num_thresh:
                            continue
                else:
                    if cell_detector is not None:
                        cell_num = simple_cell_detect(tile, cell_detector)
                        if cell_num < cell_num_thresh:
                            continue
                tile_mask = get_tile_mask(
                    tile_polygon, intersection_extract, tile_size)
                mask_name = data['maskname']
                save_que.put((tile_name, tile, mask_name, tile_mask))
        elif extract_mode == 'slide':
            if random_save:
                # do random save
                if prob > random_save_ratio:
                    continue

                tile = get_tile(slide, level_size, args)

                if is_background(tile, object_thres):
                    continue
                if cell_detector is not None:
                    cell_num = simple_cell_detect(tile, cell_detector)
                    if cell_num <= cell_num_thresh:
                        continue
                save_que.put((tile_name, tile))
            else:
                # do not random save
                tile = get_tile(slide, level_size, args)
                if is_background(tile, object_thres):
                    continue
                if cell_detector is not None:
                    cell_num = simple_cell_detect(tile, cell_detector)
                    if cell_num <= cell_num_thresh:
                        continue
                save_que.put((tile_name, tile))
    return None


def save_tile(save_que, save_mask, extract_mode):
    while True:
        data = save_que.get()
        if data is None:
            break
        if extract_mode == 'annotation':
            tile_name, tile, mask_name, mask = data

            if os.path.exists(tile_name):
                print(tile_name)
                continue

            # libpng Error: Writer error. empty disk space
            save(tile, tile_name)
            if save_mask:
                save(mask, mask_name)
            # save_tile(tile, tile_name) # Will raise OSError, No Space Left on device.
        else:
            tile_name, tile = data
            if os.path.exists(tile_name):
                print(tile_name)
                continue
            # libpng Error: Writer error. empty disk space
            save(tile, tile_name)
    return None


def save_h5(save_path, tile_names, tiles):
    h5_file = h5py.File(save_path, 'w')
    if not isinstance(tiles, np.ndarray):
        tiles = np.stack(tiles)
    if not isinstance(tile_names, np.ndarray):
        tile_names = [tmp.encode('utf8') for tmp in tile_names]
        tile_names = np.stack(tile_names)
    h5_file.create_dataset(name='tiles', data=tiles)
    h5_file.create_dataset(name='tile_names', data=tile_names)

    h5_file.close()

    return None


def aggregate_all_tiles(que):
    tile_names = []
    tiles = []
    for _ in range(que.qsize()):
        data = que.get()
        if data is None:
            break
        tile_name, tile = data
        tile_names.append(tile_name)
        tiles.append(tile)
    return tile_names, tiles


def save_tile_h5(save_path, save_que):
    tile_names, tiles = aggregate_all_tiles(save_que)
    logger.info(f'tiles save to h5 file: {len(tile_names):,}')
    save_h5(save_path, tile_names, tiles)
    return None


def tiling_for_h5(tile_que, save_que, h5_save_fp,
                  num_workers, locations, extract_coordinates,
                  exclude_coordinates, random_save, random_save_ratio,
                  extract_mode, object_thres, area_ratio):
    # TODO: this is outdated, might need to update
    tile_threads = []
    for _ in range(1):
        tile_thread = Thread(target=tile_to_que, args=(locations, tile_que))
        tile_thread.start()
        tile_threads.append(tile_thread)

    process_threads = []
    for _ in range(num_workers):
        process_thread = Thread(target=process_tile, args=(tile_que, save_que,
                                                           extract_coordinates, exclude_coordinates, random_save,
                                                           random_save_ratio, extract_mode, object_thres, area_ratio,
                                                           area_ratio))
        process_thread.start()
        process_threads.append(process_thread)

    for tile_thread in tile_threads:
        tile_thread.join()

    for _ in process_threads:
        tile_que.put(None)

    for process_thread in process_threads:
        process_thread.join()

    save_que.put(None)
    save_tile_h5(h5_save_fp, save_que)
    return None


def tiling_for_folder(tile_que, save_que, num_workers, locations, level, tile_size, overlap,
                      extract_coordinates, exclude_coordinates, random_save,
                      random_save_ratio, extract_mode, object_thres, area_ratio,
                      cell_detector, cell_num_thresh, save_mask):
    if extract_mode == 'annotation':
        process_workers = num_workers
        save_workers = num_workers
    else:
        process_workers = num_workers * 4
        if process_workers > os.cpu_count():
            process_workers = os.cpu_count() - 4
        save_workers = num_workers
    # print(f'processing workers: {process_workers}, save workers: {save_workers}')
    tile_threads = []
    for _ in range(1):
        tile_thread = Thread(target=tile_to_que, args=(locations, tile_que))
        tile_thread.start()
        tile_threads.append(tile_thread)

    process_threads = []
    for _ in range(process_workers):
        process_thread = Thread(target=process_tile, args=(tile_que, save_que, tile_size, extract_coordinates, exclude_coordinates, random_save,
                                                           random_save_ratio, extract_mode, object_thres, area_ratio, cell_detector, cell_num_thresh))
        process_thread.start()
        process_threads.append(process_thread)

    save_threads = []
    for _ in range(save_workers):
        save_thread = Thread(target=save_tile, args=(
            save_que, save_mask, extract_mode))
        save_thread.start()
        save_threads.append(save_thread)

    for tile_thread in tile_threads:
        tile_thread.join()

    for _ in process_threads:
        tile_que.put(None)

    for process_thread in process_threads:
        process_thread.join()

    for _ in save_threads:
        save_que.put(None)

    for save_thread in save_threads:
        save_thread.join()
    return None


def get_extract_exclude_coordinates(ano_file, extract_mode, extract_region_name, exclude_region_name, xml_type='imagescope'):
    if extract_mode == 'annotation':
        if not os.path.exists(ano_file):
            print("Error: annotation file not found")
        _, ano_format = os.path.splitext(ano_file)
        # here could choose paser json or xml
        if ano_format in ('.json', '.geojson'):
            coordinates, classification_names = parse_json_annotation(ano_file)
        elif ano_format in ('.xml'):
            # for different xml annotation
            if xml_type == 'imagescope':
                # imagescope xml annotation
                kwargs = dict(tag='Annotation', attrib_name='Name', tumor_group=('tumor', 'Tumor'),
                              normal_group=('normal', 'region*', None, 'None', 'none'), xml_type='imagescope')
            elif xml_type == 'asap':
                # ASAP xml annotation
                kwargs = dict(tag='Annotation', attrib_name='PartOfGroup',
                              tumor_group=tuple(('_0', '_1', 'tumor')), normal_group=('_2', 'exclusion'), xml_type='asap')
            else:
                raise NotImplementedError
            coordinates, classification_names = parse_xml_annotation(
                ano_file, **kwargs)
            # print(len(coordinates), len(classification_names))
        else:
            raise NotImplementedError

        extract_coordinates = []
        exclude_coordinates = []
        for i in range(len(classification_names)):
            names = classification_names[i]
            extract_condition = [
                True if name in extract_region_name else False for name in names]
            exclude_condition = [
                True if name in exclude_region_name else False for name in names]
            # name = classification_names[i]
            if any(extract_condition):
                extract_coordinates.append(coordinates[i])
            elif any(exclude_condition):
                exclude_coordinates.append(coordinates[i])
    elif extract_mode == 'slide':
        extract_coordinates = []
        exclude_coordinates = []
    return extract_coordinates, exclude_coordinates


def generate_slide_coordinates(shape):
    """Generates slide level coordinates for a given shape"""
    w, h = shape
    coordinates = [np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.int32)]
    return coordinates


def rescale_coordinates(slide, slide_properties, level, slide_name, extract_coordinates, exclude_coordinates, df=None):
    # TODO: as annotations is not always from level 0, so we need to calculate the right coordinates
    if df is not None:
        w, h = slide.dimensions
        selected = df[df['file_name'] == slide_name]
        # idx = df[file_names == slide_name].index[0]
        ano_width = selected['ano_width'].item()
        if w - ano_width > 2:
            ratio = w // ano_width
            extract_coordinates = [(coordinate * ratio).astype(np.int32)
                                   for coordinate in extract_coordinates]
            exclude_coordinates = [(coordinate * ratio).astype(np.int32)
                                   for coordinate in exclude_coordinates]

    # slide level extract coordinates,
    # slide.dimensions is the size o the slide's maximum magnification
    if not extract_coordinates:
        extract_coordinates = generate_slide_coordinates(slide.dimensions)

    # downsample coordinates to current level
    down_sample = int(
        float(slide_properties[f'openslide.level[{level}].downsample']))
    extract_coordinates = [np.array(
        region // down_sample).astype(np.int32) for region in extract_coordinates]
    exclude_coordinates = [np.array(
        region // down_sample).astype(np.int32) for region in exclude_coordinates]

    return extract_coordinates, exclude_coordinates


class PathExtractor(object):
    """
    Description:
        - Extract tiles from Whole-Slide-Image (WSI)

    Parameters:
        - save_path: (str), save path of all extracted tiles.
        - csv_file: (str), for qu-path annotation file, as not all annotation drawed from level 0.
        - extract_mode: (str), must but int ['slide', 'annotation'], slide is extracted tile from all slide, annotation
                        will be extracted from annotation region.
        - extract_region_name: (tuple[str]), the name of the annatotaion to extract, such as 'tumor'.
        - exclude_region_name: (tuple[str]), the name of the annocation to exclude, such as 'background'.
        - mag: (int): assume your slide is saved by 40x, 20x, 10x, 5x, 3x, 2x, 1x order. If your slide is 40x, you can get 40x, if you
                      pass 40, on the other hand, if your slide is 20x, if you pass 40, you will not get it. you must pass under 20.
        - tile_size   : (int), Which the tile size you want to split from WSI, default is 512.
        - overlap     : (int), Control the tile is overlap or not, default is 0 which mean is not overlap, all method considered
                        as non overlap, overlap tiles didnot tested if is ok.
        - save_mask: (bool), True to save the mask of tile.
        - include_bounds_size: (tuple[int, int] or int), size of the tile of slide boundary to included, if None, will be same as tile_size, which
                                will not include boundary tile, if -1 will include all. if you set as (w, h), which will not include boundary tile
                                size < w and tile size < h.
        - save_format : (string), format of the image you want to save the tiles, default is ".png".
        - object_thres: (float), determine how much object is to be considered as backgroud, default is 0.1,
                        means if less then 10% object, then will be consider as background. -1 will be save all, including background.
        - area_ratio: (tuple(float, float)), tuple of (float, float), range of area ratio between 0.0 (close) and 1.0 (open), (0.0, 1.0]
        - normalize   : (boolean), when normalize is True, will process WSI as normalized.
        - normalize_mpp: (boolean), if normalize the mpp to the same target mpp.
        - target_mpp: (float), extract tile at the same mpp, otherwise, there might be some issuse if all slides are not from only one scanner.
        - resize       : (boolean), if True, then will resize the normalize tiles to the tile_size needed, otherwise will save physical tile_size.
        - target_size  : (int), tuple, tile resize to the target size.
        - color_normalize: (bool), whether nomalized the tile or not, default is False.
        - normalize_method: (string), normalization method, default is Macenko, optional: Vahadane, Macenko, Reinhard, Ruifrok.
        - random_save: (boolean) if random save the tile or not.
        - random_save_ratio: (float) how much the tile you want to save, should be in [0.0, 1.0], 1.0 mean save all.
        - random_seed: (int) random seed of numpy.
        - test_draw_extract_regions: (boolean), if True, will draw the extract regions and save to save path, this will help to check if the
                                     extraction is correct or not, basically will help to understand the algorithm correct or not.
        - thickness: (int), thickness of the contour line.
        - scale_factor: (int), scale factor is the factor for scaling the size of the level which you extract tile down to the output image,
                               default is 64, means 64 down sample.
    """

    def __init__(self,
                 save_path: str = None,
                 csv_file: str = None,
                 extract_mode: str = None,
                 extract_region_name: Tuple[str] = None,
                 exclude_region_name: Tuple[str] = None,
                 mag: int = 40,
                 tile_size: int = 256,
                 overlap: int = 0,
                 save_mask: bool = False,
                 include_bounds_size: Tuple[int, int] = None,
                 save_format: str = 'png',
                 object_thres: float = 0.1,
                 area_ratio: tuple = (0., 1.0),
                 cell_detect: bool = False,
                 cell_detect_min_thresh=10,
                 cell_detect_max_thresh=200,
                 cell_detect_min_area=150,
                 cell_detect_max_area=1500,
                 cell_detect_min_circ=0.01,
                 cell_detect_max_circ=None,
                 cell_detect_min_conv=0.6,
                 cell_detect_max_conv=None,
                 cell_num_thresh: int = 0,
                 save_hdf5: bool = False,
                 num_workers: int = 10,
                 random_extract: bool = False,
                 sample_rate: float = 0.5,
                 iou_thresh: float = 0.5,
                 normalize_mpp: bool = False,
                 target_mpp: float = None,
                 color_normalize: bool = False,
                 random_save: bool = False,
                 random_save_ratio: float = 0.5,
                 random_seed: int = None,
                 test_draw_extract_regions: bool = False,
                 thickness: int = 2,
                 color: tuple = (0, 255, 0),
                 filled: bool = False,
                 scale_factor: int = 64) -> None:

        self.save_path = os.path.abspath(save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # where to extract the pactches from, if extract_mode is 'slide', then extract from slide, otherwise extract from annotation
        # if extract_mode is 'slide', then extract from which slide, if extract_mode is 'annotation', then extract from which annotation
        self.extract_mode = extract_mode
        assert self.extract_mode in [
            'slide', 'annotation'], f'extract mode must be "slide" or "annotation", but got {self.extract_mode}'

        if not isinstance(extract_region_name, tuple):
            extract_region_name = tuple(extract_region_name)
        self.extract_region_name = [name.lower()
                                    for name in extract_region_name]

        if not isinstance(exclude_region_name, tuple):
            exclude_region_name = tuple(exclude_region_name)
        self.exclude_region_name = [name.lower()
                                    for name in exclude_region_name]

        if self.extract_mode == 'annotation':
            len_extract_region_name = len(self.extract_region_name)
            if len_extract_region_name < 1:
                raise ValueError(
                    f'Warning: you choose annotation mode, but your extract region name has {len_extract_region_name} name.')

        self.mags = (40, 20, 10, 5, 3, 2, 1)
        assert mag in self.mags, f'mag must be one of {self.mags}, but got {mag}'
        self.mag_target = mag
        self.tile_size = tile_size
        self.overlap = overlap
        assert self.overlap < self.tile_size, f'overlap must be less than tile_size, but got overlap: {self.overlap}, tile_size: {self.tile_size}'

        self.save_mask = save_mask
        if extract_mode == 'slide':
            self.save_mask = False

        if not include_bounds_size:
            self.include_bounds_size = (tile_size, tile_size)
        elif isinstance(include_bounds_size, int) and (include_bounds_size < 0):
            self.include_bounds_size = (0, 0)
        else:
            assert isinstance(
                include_bounds_size, tuple), f'include_bounds_size must tuple, but got {type(include_bounds_size)}'
            assert len(
                include_bounds_size) == 2, f'include_bounds_size length must be 2, but got{len(include_bounds_size)}'
            self.include_bounds_size = include_bounds_size

        if not save_format.startswith('.'):
            self.save_format = '.' + save_format
        else:
            self.save_format = save_format
        self.save_format = self.save_format.lower()
        self.object_thres = object_thres

        assert area_ratio[0] >= 0, f'area ratio must be in [0.0, 1.0], but got {area_ratio}'
        assert area_ratio[1] <= 1, f'area ratio must be in [0.0, 1.0], but got {area_ratio}'
        if area_ratio[0] == 1.0:
            area_ratio = (area_ratio[0]-1e-3, area_ratio[1] + 1e-3)
        assert area_ratio[1] > area_ratio[
            0], f'area ratio lower must be smaller than upper, but got {area_ratio}'
        self.area_ratio = area_ratio

        self.save_hdf5 = save_hdf5
        self.num_workers = num_workers

        self.random_extract = random_extract
        self.sample_rate = sample_rate
        self.iou_thresh = iou_thresh
        self.normalize_mpp = normalize_mpp
        self.target_mpp = target_mpp
        self.color_normalize = color_normalize

        # set random seed
        self.random_save = random_save
        self.random_save_ratio = random_save_ratio
        if not random_seed:
            random_seed = random.randint(0, 1e6)
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.random_seed = random_seed

        self.test_draw_extract_regions = test_draw_extract_regions
        self.thickness = thickness

        if (not isinstance(color, tuple)) or len(color) != 3:
            raise ValueError(
                f'color must be tuple but got {type(color)}, and len(color) must be 3 but got {len(color)}')
        self.color = color
        self.filled = filled

        # TODO: as annotations is not always from level 0
        # csv file data is the slide level 0 shape, and annotation level shape
        # if csv_file is not None:
        #     self.df = pd.read_csv(csv_file)
        #     self.file_names = self.df['file_name']
        #     self.ano_widths = self.df['ano_width']
        # else:
        #     self.df = None
        #     self.file_names = None
        #     self.ano_widths = None
        if csv_file in (None, 'None', 'none'):
            self.df = None
        else:
            self.df = pd.read_csv(csv_file)

        if cell_detect:
            self.cell_detector = create_detector(cell_detect_min_thresh, cell_detect_max_thresh,
                                                 cell_detect_min_area, cell_detect_max_area,
                                                 cell_detect_min_circ, cell_detect_max_circ,
                                                 cell_detect_min_conv, cell_detect_max_conv)
        else:
            self.cell_detector = None
        self.cell_num_thresh = cell_num_thresh

        assert scale_factor >= 1, f'scale_factor must be greater than 1, but got {scale_factor}.'
        self.scale_factor = scale_factor

    def extract(self, file_path: str, ano_file: str, xml_type='imagescope') -> None:
        """
        Description:
            - Extract tiles from Whole-Slide-Image (WSI)

        Parameters:
            - file_path: (string), path of the WSI
        """
        slide = open_slide(file_path)
        if slide is None:
            logger.info("Error: OpenSlide failed to open the WSI")
        else:
            # get slide metrics for mpp normalization, to caculate the tile_size, also return mag
            self.slide_properties = get_slide_properties(slide)
            self.level = mag_to_level(
                self.slide_properties['mag'], self.mag_target, self.mags)
            if isinstance(self.level, int):
                self.save_dir, self.slide_name = generate_save_dir(
                    file_path, self.save_path, self.random_extract, self.level, self.tile_size)
                self.h5_save_fp = os.path.join(
                    self.save_path,  self.slide_name + '_level_' + str(self.level) + f'_{self.tile_size}' + '.h5')

                # draw extract region to show if the extract tile
                if self.test_draw_extract_regions:
                    if self.random_extract:
                        file_name = file_name = os.path.join(self.save_path, f'{self.slide_name}_{self.extract_mode}_level_'
                                                             f'{self.level}_{self.tile_size}_{self.include_bounds_size}_'
                                                             f'{self.object_thres}_{self.color}_random.png')
                    else:
                        file_name = os.path.join(self.save_path, f'{self.slide_name}_{self.extract_mode}_level_'
                                                 f'{self.level}_{self.tile_size}_{self.include_bounds_size}_'
                                                 f'{self.object_thres}_{self.color}.png')
                    if not os.path.exists(file_name):
                        logger.info(
                            f'Test mode: to check the extraction if it\'s ok ... level: {self.level}, tile_size: {self.tile_size}')
                        extract_coordinates, exclude_coordinates = get_extract_exclude_coordinates(
                            ano_file, self.extract_mode, self.extract_region_name, self.exclude_region_name, xml_type=xml_type)
                        logger.info(
                            f'extract mode: {self.extract_mode}, there are {len(extract_coordinates)} extract_coordinates and {len(exclude_coordinates)} exclude_coordinates')

                        if self.extract_mode == 'annotation':
                            if not extract_coordinates:
                                extract_coordinates = generate_slide_coordinates(
                                    slide.dimensions)

                        if self.extract_mode == 'annotation':
                            assert len(
                                extract_coordinates) > 0, f'Annotation may be not right, got 0 the extract coordinates'
                            extract_coordinates, exclude_coordinates = rescale_coordinates(slide, self.slide_properties, self.level, self.slide_name,
                                                                                           extract_coordinates, exclude_coordinates, self.df)

                        # Get tile locations
                        if self.random_extract:
                            # Random tile coordinates
                            self.locations, num_tiles_width, num_tiles_height = random_generate_tile_locations(slide, self.slide_properties, self.level, self.tile_size,
                                                                                                               self.save_dir, self.extract_mode, self.slide_name, self.save_format,
                                                                                                               extract_coordinates, self.sample_rate, self.iou_thresh)
                        else:
                            # Grid tile coordinates
                            self.locations, num_tiles_width, num_tiles_height = get_tile_location(slide, self.slide_properties, self.level, self.tile_size, self.overlap,
                                                                                                  self.save_dir, self.slide_name, self.save_format, self.include_bounds_size)

                        if not isinstance(self.locations, list):
                            self.locations = list(self.locations)
                        logger.info(
                            f'locations: {len(self.locations):,}, (m, n): ({num_tiles_width}, {num_tiles_height})')

                        draw_extract_regions(file_path, self.slide_properties, file_name, self.num_workers, self.level, self.locations,
                                             extract_coordinates, exclude_coordinates, self.extract_mode, self.object_thres,
                                             self.area_ratio, self.random_save, self.random_save_ratio, self.cell_detector,
                                             self.cell_num_thresh, self.thickness, self.color, self.filled, self.scale_factor)
                    else:
                        logger.warning(f'{file_name} already exist...')
                else:
                    tile_que = Queue(maxsize=5000)
                    save_que = Queue()

                    # if save directory not exists, and not save_hdf5 will extract the slide
                    # if h5 save file path not exits, and save_hdf5 is True, then will extract and save to h5
                    # else if save_hdf5 or not, if not then will check the directory if empty or not, if empty will skip this slide
                    # else will extract tile to directory
                    P, Q, R = self.get_process_condition()
                    if (not P and not R) or (not Q and R):
                        # processing
                        logger.info(f'Processing {self.slide_name}...')
                        # here we can get the slice annotation and selecte which region to be extracted from
                        extract_coordinates, exclude_coordinates = get_extract_exclude_coordinates(ano_file, self.extract_mode,
                                                                                                   self.extract_region_name, self.exclude_region_name, xml_type=xml_type)
                        logger.info(
                            f'extract mode: {self.extract_mode}, there are {len(extract_coordinates)} extract_coordinates and {len(exclude_coordinates)} exclude_coordinates')

                        # make sure the extract & exclude coordinates are in same ratio with level of slide
                        if self.extract_mode == 'annotation':
                            assert len(
                                extract_coordinates) > 0, f'Annotation may be not right, got 0 the extract coordinates'
                            extract_coordinates, exclude_coordinates = rescale_coordinates(slide, self.slide_properties, self.level, self.slide_name,
                                                                                           extract_coordinates, exclude_coordinates, self.df)
                         # Get tile locations
                        if self.random_extract:
                            # Random tile coordinates
                            self.locations, num_tiles_width, num_tiles_height = random_generate_tile_locations(slide, self.slide_properties, self.level, self.tile_size,
                                                                                                               self.save_dir, self.extract_mode, self.slide_name, self.save_format,
                                                                                                               extract_coordinates, self.sample_rate, self.iou_thresh)
                        else:
                            # Grid tile coordinates
                            self.locations, num_tiles_width, num_tiles_height = get_tile_location(slide, self.slide_properties, self.level, self.tile_size, self.overlap,
                                                                                                  self.save_dir, self.slide_name, self.save_format, self.include_bounds_size)

                        if not isinstance(self.locations, list):
                            self.locations = list(self.locations)
                        logger.info(
                            f'tile size {self.tile_size}, overlap: {self.overlap}, locations: {len(self.locations):,}, (m, n): ({num_tiles_width}, {num_tiles_height})')

                        # turn coordinates to polygon, add buffer avoid errors
                        extract_coordinates = geometry.MultiPolygon(
                            polygon_to_multipolygon(extract_coordinates)).buffer(0.01)
                        exclude_coordinates = geometry.MultiPolygon(
                            polygon_to_multipolygon(exclude_coordinates)).buffer(0.01)
                        if self.save_hdf5:
                            if not os.path.exists(self.h5_save_fp):
                                logger.info('tiling for h5...')
                                tiling_for_h5(tile_que, save_que, self.h5_save_fp, self.num_workers+2, self.locations,
                                              self.deep_gen, self.deep_level, extract_coordinates, exclude_coordinates,
                                              self.random_save, self.random_save_ratio, self.extract_mode, self.object_thres)
                            else:
                                logger.warning(
                                    f'{os.path.split(self.h5_save_fp)[-1]} already exists...')
                        else:
                            os.makedirs(self.save_dir, exist_ok=True)
                            if self.save_mask:
                                # create directory for masks files
                                os.makedirs(os.path.join(os.path.dirname(
                                    self.save_dir), 'masks'), exist_ok=True)
                            logger.info(f'tiling {self.slide_name}...')
                            tiling_for_folder(tile_que, save_que, self.num_workers, self.locations, self.level, self.tile_size, self.overlap,
                                              extract_coordinates, exclude_coordinates, self.random_save,
                                              self.random_save_ratio, self.extract_mode, self.object_thres, self.area_ratio,
                                              self.cell_detector, self.cell_num_thresh, self.save_mask)
                    else:
                        if self.save_hdf5:
                            logger.warning(
                                f'{os.path.split(self.h5_save_fp)[-1]} already exists..')
                        else:
                            if not os.listdir(self.save_dir):
                                if self.save_mask:
                                    # create directory for masks files
                                    os.makedirs(os.path.join(os.path.dirname(
                                        self.save_dir), 'masks'), exist_ok=True)
                                # processing
                                logger.warning(
                                    'directory exits but directory is empty, processing for folder')
                                # here we can get the slide annotation and selecte which region to be extracted from
                                extract_coordinates, exclude_coordinates = get_extract_exclude_coordinates(ano_file, self.extract_mode,
                                                                                                           self.extract_region_name, self.exclude_region_name, xml_type=xml_type)
                                logger.info(
                                    f'extract mode: {self.extract_mode}, there are {len(extract_coordinates)} extract_coordinates and {len(exclude_coordinates)} exclude_coordinates')

                                # make sure the extract & exclude coordinates are in same ratio with level of slide
                                if self.extract_mode == 'annotation':
                                    extract_coordinates, exclude_coordinates = rescale_coordinates(slide, self.slide_properties, self.level, self.slide_name,
                                                                                                   extract_coordinates, exclude_coordinates, self.df)

                                 # Get tile locations
                                if self.random_extract:
                                    # Random tile coordinates
                                    self.locations, num_tiles_width, num_tiles_height = random_generate_tile_locations(slide, self.slide_properties, self.level, self.tile_size,
                                                                                                                       self.save_dir, self.extract_mode, self.slide_name, self.save_format,
                                                                                                                       extract_coordinates, self.sample_rate, self.iou_thresh)
                                else:
                                    # Grid tile coordinates
                                    self.locations, num_tiles_width, num_tiles_height = get_tile_location(slide, self.slide_properties, self.level, self.tile_size, self.overlap,
                                                                                                          self.save_dir, self.slide_name, self.save_format, self.include_bounds_size)
                                logger.info(
                                    f'tile size {self.tile_size}, overlap: {self.overlap}, locations: {len(self.locations):,}, (m, n): ({num_tiles_width}, {num_tiles_height})')

                                # turn coordinates to polygon, add buffer avoid errors
                                # extract_coordinates = [geometry.Polygon(coord.squeeze(0)).buffer(0.01) for coord in extract_coordinates if (coord is not None and coord.shape[1] > 2)]
                                # exclude_coordinates = [geometry.Polygon(coord.squeeze(0)).buffer(0.01) for coord in exclude_coordinates]
                                extract_coordinates = geometry.MultiPolygon(
                                    polygon_to_multipolygon(extract_coordinates)).buffer(0.01)
                                exclude_coordinates = geometry.MultiPolygon(
                                    polygon_to_multipolygon(exclude_coordinates)).buffer(0.01)
                                logger.info(f'tiling {self.slide_name}...')
                                tiling_for_folder(tile_que, save_que, self.num_workers, self.locations, self.level, self.tile_size, self.overlap,
                                                  extract_coordinates, exclude_coordinates, self.random_save,
                                                  self.random_save_ratio, self.extract_mode, self.object_thres, self.area_ratio,
                                                  self.cell_detector, self.cell_num_thresh, self.save_mask)
                            else:
                                logger.warning(
                                    f'{os.path.split(self.save_dir)[-1]} already processed...')
            else:
                logger.warning(f'level is None, this slide will be skipped...')

    def get_process_condition(self):
        P = os.path.exists(self.save_dir)
        Q = os.path.exists(self.h5_save_fp)
        R = self.save_hdf5
        return P, Q, R


if __name__ == '__main__':
    time_message = get_localtime()
    logger.set_formatter(None)
    logger.info(time_message.center(120, '*'))
    logger.reset_default_formatter()

    csv_file = None
    save_path = './data/extract_patches_test'
    kwargs = dict(save_path=save_path,
                  csv_file=csv_file,
                  extract_mode='slide',
                  extract_region_name=('tumor', ),
                  exclude_region_name=('normal',),
                  mag=20,
                  patch_size=256,
                  overlap=0,
                  save_mask=True,
                  include_bounds_size=None,
                  save_format='png',
                  object_thres=0.1,
                  area_ratio=(0.3, 1.0),
                  cell_detect=False,
                  cell_detect_min_thresh=10,
                  cell_detect_max_thresh=200,
                  cell_detect_min_area=150,
                  cell_detect_max_area=1500,
                  cell_detect_min_circ=0.01,
                  cell_detect_max_circ=None,
                  cell_detect_min_conv=0.6,
                  cell_detect_max_conv=None,
                  cell_num_thresh=120,
                  # 2023-08-24 update, 1 is best performance for annotation mode, vary by tile size, 4 is best for slide mode
                  num_workers=16,
                  save_hdf5=False,
                  random_extract=True,
                  sample_rate=2,
                  iou_thresh=0.2,
                  normalize_mpp=False,
                  target_mpp=None,
                  color_normalize=False,
                  random_save=False,
                  random_save_ratio=0.6,
                  random_seed=4,
                  test_draw_extract_regions=True,
                  thickness=1,
                  # navy (0, 0, 128), medium blue(0, 0, 205), light green (125, 220, 0), red (238, 0, 0)
                  color=(0, 0, 205),
                  filled=True,
                  scale_factor=32)
    extractor = PathExtractor(**kwargs)
    logger.set_level(2, 0)
    # logger.info(f'Start extracting tiles from directory {wsi_path}')
    logger.info(f'Extractor kwargs: {extractor.__dict__}')
    logger.set_level(1, 0)

    # single slide test
    # CMU
    filename = r'/mnt/f/Work/Data/CMU/CMU-1.svs'
    ano_path = r'/mnt/f/Work/Data/CMU/'

    dir, file_name = get_dir_filename(filename)
    ano_file = os.path.join(ano_path, file_name.split('.')[0] + '.xml')

    start_time = time.time()
    extractor.extract(filename, ano_file, xml_type='imagescope')
    end_time = time.time()
    print('Done!!! time: {}'.format(end_time - start_time))
