import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
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
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# SAMPLE_IMAGE_DIR = os.path.join(ROOT_DIR, "data", "head_crops_sample")
BATCH_SIZE = 1

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def getImages(image_dir, image_group):
    images = []
    for image_name in image_group:
        img_path = os.path.join(image_dir, image_name)
        if os.path.exists(img_path) and (img_path.find('.jpg') != -1 or img_path.find('.jpeg') != -1 or img_path.find('.png') != -1):
            image = skimage.io.imread(os.path.join(image_dir, image_name))
            images.append(image)
        else:
            print('image path does not exist', img_path)
    if len(images) == 0:
        return None
    return images

def formatOutName(image_name):
    name = '.'.join((image_name.strip().split('.'))[0:-1]) + '.json'
    return name

def objectDet(image_dir, out_dir, heads):
    batch_size = config.BATCH_SIZE
    image_names = os.listdir(image_dir)
    format_name = '.' + (image_names[0].strip().split('.'))[-1]
    print('format: ', format_name)
    already_done = os.listdir(out_dir)
    already_done = set(['.'.join((name.strip().split('.'))[0:-1]) + format_name for name in already_done])
    print('already done: ', len(already_done))
    image_names = [ img_name for img_name in image_names if (img_name in heads and img_name not in already_done)]
    print('images left to detect: ', len(image_names))
    for image_group in chunker(image_names, batch_size):
        images = getImages(image_dir, image_group)
        if images is not None:
            diff = batch_size - len(images)
            if diff != 0:
                for k in range(diff):
                    images.append(images[-1]) # append last image k times
            results = model.detect(images, verbose=0)
            assert len(results) == len(images)

            for i in range((batch_size-diff)): # ith image
                image_name = image_group[i]
                json_data = {}
                result = results[i]
                json_dets = []
                class_ids = result['class_ids']
                bboxes = result['rois']
                scores = result['scores']
                image_path = os.path.join(image_name, image_name)
                for j in range(len(class_ids)): # jth detection
                    json_det = {}
                    bbox = [int(x) for x in bboxes[j]]
                    json_det['bbox'] = [bbox[1], bbox[0], bbox[3], bbox[2]]
                    json_det['score'] = float(scores[j])
                    json_det['class'] = class_names[int(class_ids[j])]
                    json_dets.append(json_det)
                    # print(json_det)
                json_data['path'] = image_path
                json_data['detections'] = json_dets

                json_out_path = os.path.join(out_dir, formatOutName(image_name))
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                print("Saving ", image_name, ' --> ', json_out_path)
                with open(json_out_path, 'w') as f:
                    json.dump(json_data, f)

def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description='MaskRCNN object detector printer')
    parser.add_argument('--images', type=str,
                        help='Path to directory containing images', required=True)
    parser.add_argument('--outdir', type=str,
                        help='Path to output directory', required=True)
    parser.add_argument('--head',  type=str,
                        help='Path to csv containing head annotations', required=True)

    # parser.add_argument('--batchsize', default=2, type=int,
    #                     help='Path to output directory', required=False)

    global args
    args = parser.parse_args(argv)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

    IMAGES_PER_GPU = BATCH_SIZE
    GPU_COUNT = 1



if __name__ == '__main__':
    parseArgs()

    # Initialize config
    config = InferenceConfig()
    config.display()

    # Load weights
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Load COCO dataset
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, "train")
    dataset.prepare()
    class_names = dataset.class_names
    print(class_names)

    # Detect objects
    image_dir = args.images
    out_dir = args.outdir
    head_file = args.head
    heads = open(head_file, 'r').readlines()
    heads = [(((h.strip().split('\t'))[0]).split('/'))[-1] for h in heads]
    print(heads[0:10])
    heads = set(heads)
    print('heads: ', len(heads))
    objectDet(image_dir, out_dir, heads)
