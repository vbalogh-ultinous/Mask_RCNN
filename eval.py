import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import skimage.io
import json

ROOT_DIR = os.path.abspath(".")
DATA_DIR = "/data/vbalogh/Mask_RCNN/data"
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(DATA_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
COCO_DIR = "/data/vbalogh/coco"
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# SAMPLE_IMAGE_DIR = os.path.join(ROOT_DIR, "data", "head_crops_sample")
BATCH_SIZE=2

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = BATCH_SIZE

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Load COCO dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()

# Print class names
class_names = dataset.class_names
print(class_names)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def objectDet(image_names=None, batch_size=config.BATCH_SIZE):
    if image_names==None:
        image_names = os.listdir(IMAGE_DIR)[0:30]

    for image_group in chunker(image_names, batch_size):
        images = []
        for image_name in image_group:
            image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))
            images.append(image)
        results = model.detect(images, verbose=0)
        assert len(results) == len(images)
        for i in range(len(results)): # ith image
            json_data = {}
            result = results[i]
            json_dets = []
            class_ids = result['class_ids']
            bboxes = result['rois']
            scores = result['scores']
            print(class_ids)
            image_path = os.path.join(IMAGE_DIR, image_group[i])
            for j in range(len(class_ids)): # jth detection
                json_det = {}
                json_det['bbox'] = bboxes[j]
                json_det['score'] = scores[j]
                json_det['class'] = class_ids[j]
                json_dets.append(json_det)
                print(json_det)
            json_data['path'] = image_path
            json_data['detections'] = json_dets
        json_out_path = '/data/vbalogh/Mask_RCNN/data/' + image_name + '.json'
        with open(json_out_path, 'w') as f:
            json.dump(json_data, f)




objectDet()