#!/bin/bash
HEAD=${1-/data/vbalogh/head_det_corpus_v3/bounding_boxes/train_v3.csv}
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film9 --outdir /data/vbalogh/Mask_RCNN/detections/film9 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film8 --outdir /data/vbalogh/Mask_RCNN/detections/film8 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film7 --outdir /data/vbalogh/Mask_RCNN/detections/film7 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film6 --outdir /data/vbalogh/Mask_RCNN/detections/film6 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film5 --outdir /data/vbalogh/Mask_RCNN/detections/film5 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film4 --outdir /data/vbalogh/Mask_RCNN/detections/film4 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film3 --outdir /data/vbalogh/Mask_RCNN/detections/film3 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film2 --outdir /data/vbalogh/Mask_RCNN/detections/film2 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film1 --outdir /data/vbalogh/Mask_RCNN/detections/film1 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/rossmann_cash1 --outdir /data/vbalogh/Mask_RCNN/detections/rossmann_cash1 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/rossmann_cash2 --outdir /data/vbalogh/Mask_RCNN/detections/rossmann_cash2 --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/rossmann_entrance --outdir /data/vbalogh/Mask_RCNN/detections/rossmann_entrance --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/rossmann_line --outdir /data/vbalogh/Mask_RCNN/detections/rossmann_line --head $HEAD

python3 eval.py --images /data/vbalogh/head_det_corpus_v3/HollywoodHeads/JPEGImages --outdir /data/vbalogh/Mask_RCNN/detections/HollywoodHeads/JPEGImages --head $HEAD
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/MPII/images --outdir /data/vbalogh/Mask_RCNN/detections/MPII/images --head $HEAD
