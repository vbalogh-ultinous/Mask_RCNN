#!/bin/bash

HEAD=${1-/home/vbalogh/personDet/head_det_v3_bounding_boxes/train_v3.csv}
PERSON=${2-/data/vbalogh/yolact/detections/head_det_corpus_v3/film8}
IMAGES=${3-/data/vbalogh/head_det_corpus_v3/film8}
OUTDIR=${4-/data/vbalogh/yolact/align/film8}
python align.py --head $HEAD --person $PERSON --images $IMAGES --outdir $OUTDIR
