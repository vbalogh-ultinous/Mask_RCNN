import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import argparse

def computeIoU(head_bb, person_bb, epsilon=0.1, threshold=0.7):
    """
    compute the ratio of intersection and union of the given head and the given person
    the area of the person is weighted by epsilon
    intersection over union = area of overlap / (area of head-box + epsilon * area of body-box)
    :param person_bb: person bounding box
    :param head_bb: head bounding box
    :param epsilon: weight for person area
    :return: "intersection over union"-like stuff
    """
    headbox_area = (head_bb[2]-head_bb[0])*(head_bb[3]-head_bb[1])
    person_area = (person_bb[2]-person_bb[0])*(person_bb[3]-person_bb[1])
    dx = min(head_bb[2], person_bb[2])-max(head_bb[0], person_bb[0])
    dy = min(head_bb[3], person_bb[3])-max(head_bb[1], person_bb[1])
    result = 0
    overlap_area = 0
    if dx > 0 and dy > 0: # make sure person and head intersects
        overlap_area = dx * dy
    if computeIoH(overlap_area, headbox_area) > threshold: # TODO max problem instead of min
        result = -overlap_area / (headbox_area + epsilon * person_area)
    # if np.abs(result) > threshold:
    #     return result
    # else:
    #     return 0
    return result

def computeIoH(overlap, head):
    """
    compute the ratio of intersection (of head and person) and head area
    intersection over head-box = area of overlap / area of head-box
    :param overlap: area of intersection
    :param head: area of head
    :return: IoH
    """
    return overlap/head


# in progress...
def center(person_bb, head_bb, distance='euclidean'):
    # compute distance from the two centers
    width_head = head_bb[2]-head_bb[0]
    height_head = head_bb[3]-head_bb[1]
    center_head = np.array([head_bb[0]+width_head/2, head_bb[1]+height_head/2])
    width_person = person_bb[2]-person_bb[0]
    height_person = person_bb[3]-person_bb[3]
    center_person = np.array([person_bb[0]+width_person/2, person_bb[1]+height_person/2])
    return distance

def generateColor():
    """
    random generate a colour
    :return: random GBR color
    """
    color = tuple(np.random.choice(range(256), size=3))
    return tuple(int(c) for c in color)

def getSuffix(person_dir):
    suffix = (person_dir.strip().split('/'))[-1]
    if suffix == '':
        suffix = (person_dir.strip().split('/'))[-2]
    return suffix

def makeOutDir(person_dir, out_dir_path):
    if out_dir_path == None:
        suffix = getSuffix(person_dir)
        out_dir_path = os.path.join('results/match', suffix)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    return out_dir_path

def getMismatchedIndices(bboxes, aligned_indices):
    """
    compute the indices of the bounding boxes
    that do not appear in any of the head-person pairs (matched by the hungarian algorithm)
    :param bboxes: bounding boxes
    :param aligned_indices: matched indices of bounding boxes
    :return: list of indices (of bounding boxes) that are not matched
    """
    return [i for i in range(len(bboxes)) if i not in aligned_indices]


def drawRectangles(indices, C, head_bbs, person_bbs, image):
    """
    draw head and body bounding boxes on image
    :param indices: indices of the paired head bounding boxes and body bounding boxes
    :param C: cost matrix
    :param head_bbs: head bounding boxes
    :param person_bbs: person bounding boxes
    :param image: image to draw the rectangles on
    """
    pair_indices = [(ind1, ind2) for ind1, ind2 in zip(indices[0], indices[1])]
    for (row_ind, col_ind) in pair_indices:
        if C[row_ind, col_ind] < 0:
            # print('Head: ', row_ind, head_bbs[row_ind], '\nPerson: ', col_ind, person_bbs[col_ind])
            color = generateColor()
            cv2.rectangle(image, (head_bbs[row_ind][0], head_bbs[row_ind][1]),
                          (head_bbs[row_ind][2], head_bbs[row_ind][3]),
                          color, 2)
            cv2.rectangle(image, (person_bbs[col_ind][0], person_bbs[col_ind][1]),
                          (person_bbs[col_ind][2], person_bbs[col_ind][3]),
                          color, 1)
        else:
            (indices[0].tolist()).remove(row_ind)
            (indices[1].tolist()).remove(col_ind)
    for i in getMismatchedIndices(head_bbs, indices[0]):
        cv2.rectangle(image, (head_bbs[i][0], head_bbs[i][1]), (head_bbs[i][2], head_bbs[i][3]),
                      (0, 0, 255), 2)
    for i in getMismatchedIndices(person_bbs, indices[1]):
        cv2.rectangle(image, (person_bbs[i][0], person_bbs[i][1]), (person_bbs[i][2], person_bbs[i][3]),
                      (0, 255, 0), 1)

def getPersonBoundingBoxes(person_dir, filename):
    json_data = json.load(open(os.path.join(person_dir, filename)))
    detections = []
    if 'detections' in json_data.keys():
        detections = json_data['detections']
    person_bbs = [det['bbox'] for det in detections if det['class'] == 'person']
    person_bbs = [[person_bb[1], person_bb[0], person_bb[3], person_bb[2]] for person_bb in person_bbs]
    return person_bbs

def getHeadBoundingBoxes(head_file, person_dir, filename):
    heads = open(head_file, 'r').readlines()
    raw_filename = (person_dir.strip().split('/'))[-1] + '/' + '.'.join((filename.strip().split('.'))[0:-1])
    head_line = [line for line in heads if line.find(raw_filename) != -1]
    # print(raw_filename, head_line, person_bbs)
    head_bbs = []
    if len(head_line) > 0:  # and len(person_bbs) > 0:
        head_bbs = (head_line[0].strip().split('\t'))[1:]
        head_bbs = [[int(head_bbs[i]), int(head_bbs[i + 1]), int(head_bbs[i + 2]), int(head_bbs[i + 3])] for i
                    in range(len(head_bbs)) if i % 5 == 0]
    return head_bbs

def computeAlginments(head_bbs, person_bbs):
    indices = np.array([[], []])
    C = np.zeros([len(head_bbs), len(person_bbs)])
    if len(head_bbs) > 0 and len(person_bbs) > 0:
        C = cdist(XA=np.array(head_bbs), XB=np.array(person_bbs), metric=computeIoU)  # maximize
        indices = linear_sum_assignment(C)
    return indices, C

def computeMetrics(C, aligned_indices, head_bbs, person_bbs, cummulated_metrics):
    # remove indices from matched indices that have 0 cost, since they are not a real match
    pair_indices = [(ind1, ind2) for ind1, ind2 in zip(aligned_indices[0], aligned_indices[1])]
    for (row_ind, col_ind) in pair_indices:
        if C[row_ind, col_ind] >= 0:
            (aligned_indices[0].tolist()).remove(row_ind)
            (aligned_indices[1].tolist()).remove(col_ind)
    cost = - C[aligned_indices[0], aligned_indices[1]].sum()/float(len(aligned_indices[0]))

    heads = len(head_bbs)
    mismatched_heads = len(getMismatchedIndices(head_bbs, aligned_indices[0]))
    matched_head_ratio = (heads - mismatched_heads) / heads
    people = len(person_bbs)
    mismatched_people = len(getMismatchedIndices(person_bbs, aligned_indices[1]))
    matched_person_ratio = (people - mismatched_people) / people
    matched_objects_ratio = len(aligned_indices[0])/ float(len(aligned_indices[0]) + mismatched_people + mismatched_heads)

    cummulated_metrics['count'] += 1
    cummulated_metrics['cost'] += cost
    cummulated_metrics['matched_head_ratio'] += matched_head_ratio
    cummulated_metrics['matched_person_ratio'] += matched_person_ratio
    cummulated_metrics['matched_object_ratio'] += matched_objects_ratio
    print(cost, matched_head_ratio, matched_person_ratio, matched_objects_ratio)

def finalizeMetrics(cummulated_metrics):
    metrics = {'count': 0, 'cost': 0, 'matched_head_ratio': 0.0, 'matched_person_ratio': 0.0, 'matched_object_ratio': 0.0}
    count = cummulated_metrics['count']
    metrics['count'] = count
    metrics['cost'] = cummulated_metrics['cost']/float(count)
    metrics['matched_head_ratio'] = cummulated_metrics['matched_head_ratio']/float(count)
    metrics['matched_person_ratio'] = cummulated_metrics['matched_person_ratio']/float(count)
    metrics['matched_object_ratio'] = cummulated_metrics['matched_object_ratio']/float(count)
    return metrics

def Align(head_file, person_dir, image_dir, out_dir):
    # heads = open(head_file, 'r').readlines()
    # heads.extend(open(HEAD_bb_path_2, 'r').readlines())
    print('Reading in files')
    cummulated_metrics = {'count': 0, 'cost': 0, 'matched_head_ratio': 0.0, 'matched_person_ratio': 0.0, 'matched_object_ratio': 0.0}
    for filename in os.listdir(person_dir):
        if filename.find('.json') != -1:
            person_bbs = getPersonBoundingBoxes(person_dir, filename)
            head_bbs = getHeadBoundingBoxes(head_file, person_dir, filename)
            indices, C = computeAlginments(head_bbs, person_bbs)
            img_format = '.png'
            if image_dir.find('HollywoodHeads')!=-1:
                img_format = '.jpeg'
            elif image_dir.find('MPII')!=-1:
                img_format = '.jpg'
            img_filename = '.'.join((filename.strip().split('.'))[0:-1]) + img_format
            image = cv2.imread(os.path.join(image_dir, img_filename))
            drawRectangles(indices, C,  head_bbs, person_bbs, image)
            print(img_filename, ' --> ', os.path.join(out_dir, img_filename))
            cv2.imwrite(os.path.join(OUT_DIR, img_filename), image)
            computeMetrics(C, indices, head_bbs, person_bbs, cummulated_metrics)
    metrics = finalizeMetrics(cummulated_metrics)
    print(metrics)

def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description='(MaskRCNN) body-head aligner')
    parser.add_argument('--head',
                        default='results/head_bounding_boxes/train_v3.csv', type=str,
                        help='Path to annotated head bounding boxes csv file', required=False)
    parser.add_argument('--person', default='results/person_bounding_boxes/film8', type=str,
                        help='Path to (yolact) directory containing person bounding boxes jsons', required=False)
    parser.add_argument('--images', default='data/head_det_corpus_v3/film8', type=str,
                        help='Path to directory containing raw images', required=False)
    parser.add_argument('--outdir', default=None, type=str,
                        help='Path to output directory', required=False)

    global args
    args = parser.parse_args(argv)

if __name__ == '__main__':
    parseArgs()

    HEAD_FILE = args.head
    PERSON_DIR = args.person
    IMAGE_DIR = args.images
    OUT_DIR = makeOutDir(PERSON_DIR, args.outdir)

    Align(HEAD_FILE, PERSON_DIR, IMAGE_DIR, OUT_DIR)
