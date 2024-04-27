# PathEx: A Python-based Algorithm for Extracting Whole Slide Image Tiles

This Python-based tool extracts tiles from whole slide images (WSIs). It supports both grid-based and random extraction of tiles, and includes options for cell detection, color normalization, and saving the extracted tiles in different formats.

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Slide level Grid-based extraction:

images are extracted in a grid pattern, with a specified tile size. Argments are as follows:
```bash
extract_mode='slide', mag=10,tile_size=512, overlap=0, save_format='png', object_thresh=0.1, area_ration=(0.3, 1.0), random_extract=False, test_draw_extract_regions=True, scale_factor=32
```
<img src="images/CMU-1_slide_level_1_512_(512, 512)_0.1_(0, 0, 205).png" width="500" />

### Slide level Random extraction:
Tile images are extracted randomly.
```bash
extract_mode='slide', mag=10, tile_size=256, overlap=0, save_format='png', object_thresh=0.1, area_ration=(0.3, 1.0), random_extract=True, test_draw_extract_regions=True, scale_factor=32
```
<img src="images/CMU-1_slide_level_1_256_(256, 256)_0.1_(0, 0, 205)_random.png" width="500" />

### Annotation extraction:
To extract tiles from annotations, you need to provide the annotation file path and the corresponding WSI file path. The annotation file should be in the same format as the one used by the Aperio ImageScope software or the QuPath software. The tile extraction will be done based on the annotation coordinates. We recommend using the Aperio ImageScope software to create the annotation file.

```bash
extract_mode='slide', mag=10, tile_size=256, overlap=0, save_format='png', object_thresh=0.1, area_ration=(0.3, 1.0), random_extract=False, test_draw_extract_regions=True, scale_factor=32
```
<img src="images/CMU-1_annotation_level_1_256_(256, 256)_0.1_(0, 0, 205).png" width="500" />


### Annotation random extraction:
You can also randomly extract tiles from annotations.
```bash
extract_mode='slide', mag=10, tile_size=256, overlap=0, save_format='png', object_thresh=0.1, area_ration=(0.3, 1.0), random_extract=True, test_draw_extract_regions=True, scale_factor=32
```
<img src="images/CMU-1_annotation_level_1_256_(256, 256)_0.1_(0, 0, 205)_random.png" width="500" />

### Prerequisites

The project requires Python 3.9 and the following Python libraries:

- openslide-python
- numpy
- opencv-python
- tifffile
- shapely

You can install these libraries using conda:

```bash
conda env create --file requirements.yml
```

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
conda env create --file requirements.yml
```

## Usage
Run the script with the desired arguments:

```bash
python main.py --wsi-path /path/to/wsi --save-path /path/to/save --tile-size 512
```

or you can run by shell script:

```bash
bash run.sh
```

## Some problems may occur

1. OSError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```
2. cannot import name 'Feature' from 'setuptools'

```bash
pip install --upgrade pip setuptools
or
pip install setuptools==57.5.0
```

## Contributing
Contributions are welcome! Please read the contributing guidelines before getting started.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this tool in your research, please cite the following paper:

```
@article{patchex,
  title={PatchEx: A Python-based Tool for Extracting Whole Slide Image Tiles},
  author={<NAME> and <NAME> and <NAME> and <NAME> and <NAME>},
  journal={arXiv preprint arXiv:2203.16155},
  year={2022}
}
```