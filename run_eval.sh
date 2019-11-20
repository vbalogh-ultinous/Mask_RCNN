#!/bin/bash

python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film9 --outdir /data/vbalogh/MaskRCNN/detections/film9 
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film8 --outdir /data/vbalogh/MaskRCNN/detections/film8 
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film7 --outdir /data/vbalogh/MaskRCNN/detections/film7 
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film6 --outdir /data/vbalogh/MaskRCNN/detections/film6 
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film5 --outdir /data/vbalogh/MaskRCNN/detections/film5 
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film4 --outdir /data/vbalogh/MaskRCNN/detections/film4 
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film3 --outdir /data/vbalogh/MaskRCNN/detections/film3 
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film2 --outdir /data/vbalogh/MaskRCNN/detections/film2 
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/film1 --outdir /data/vbalogh/MaskRCNN/detections/film1
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/rossmann_cash1 --outdir /data/vbalogh/MaskRCNN/detections/rossmann_cash1
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/rossmann_cash2 --outdir /data/vbalogh/MaskRCNN/detections/rossmann_cash2
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/rossmann_entrance --outdir /data/vbalogh/MaskRCNN/detections/rossmann_entrance
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/rossmann_line --outdir /data/vbalogh/MaskRCNN/detections/rossmann_line

python3 eval.py --images /data/vbalogh/head_det_corpus_v3/HollywoodHeads/JPEGImages --outdir /data/vbalogh/MaskRCNN/detections/HollywoodHeads/JPEGImages
python3 eval.py --images /data/vbalogh/head_det_corpus_v3/MPII/images --outdir /data/vbalogh/MaskRCNN/detections/MPII/images
