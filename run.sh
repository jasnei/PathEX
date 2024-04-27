#!bin/bash
python main.py \
    --wsi-path=your_wsi_path \
    --wsi-suffix=tif \
    --recursive=True \
    --save-path=your_save_path \
    --ano-path=your_ano_path \
    --ano-format=.xml \
    --xml-type=imagescope \
    --csv-file=None \
    --extract-mode=slide \
    --extract-region-name=tumor \
    --mag=40 \
    --tile-size=1024\
    --overlap=0 \
    --save-format=png \
    --object-thres=0.15 \
    --include-boundary=False \
    --save-hdf5=False \
    --num-workers=16 \
    --random-save=True \
    --random-save-ratio=0.3 \
    --random-seed=42 \