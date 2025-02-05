import numpy as np
import json
import os
import cv2
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
        result = - overlap_area / (headbox_area + epsilon * person_area)
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

def isCovered(hx1, hy1, hx2, hy2, bx1, by1, bx2, by2):
    if( hx1 < bx1 or bx2 < hx2 or hy1 < by1 or by2 < hy2):
        return False
    return True

def drawRectangles(indices, C, head_bbs, person_bbs, image, image_file_name):
    """
    draw head and body bounding boxes on image
    :param indices: indices of the paired head bounding boxes and body bounding boxes
    :param C: cost matrix
    :param head_bbs: head bounding boxes
    :param person_bbs: person bounding boxes
    :param image: image to draw the rectangles on
    :param image_file_name: path to image
    """
    text = []
    text.append(image_file_name)
    cover_ratio = 0.0
    pair_indices = [(ind1, ind2) for ind1, ind2 in zip(indices[0], indices[1])]
    for (row_ind, col_ind) in pair_indices:
        if C[row_ind, col_ind] < 0:
            color = generateColor()
            cv2.rectangle(image, (head_bbs[row_ind][0], head_bbs[row_ind][1]),
                          (head_bbs[row_ind][2], head_bbs[row_ind][3]),
                          color, 2)
            cv2.rectangle(image, (person_bbs[col_ind][0], person_bbs[col_ind][1]),
                          (person_bbs[col_ind][2], person_bbs[col_ind][3]),
                          color, 1)
            indices_text = '1\t' + str(head_bbs[row_ind][0]) + '\t' + str(head_bbs[row_ind][1]) \
                      + '\t' + str(head_bbs[row_ind][2]) + '\t' + str(head_bbs[row_ind][3]) \
                      + '\t' + str(person_bbs[col_ind][0]) + '\t' + str(person_bbs[col_ind][1]) \
                      + '\t' + str(person_bbs[col_ind][2]) + '\t' + str(person_bbs[col_ind][3])
            text.append(indices_text)
            # compute cover ratio
            if(isCovered(head_bbs[row_ind][0], head_bbs[row_ind][1], head_bbs[row_ind][2], head_bbs[row_ind][3],
               person_bbs[col_ind][0], person_bbs[col_ind][1], person_bbs[col_ind][2], person_bbs[col_ind][3])):
                cover_ratio += 1.0
        else:
            (indices[0].tolist()).remove(row_ind)
            (indices[1].tolist()).remove(col_ind)
    for i in getMismatchedIndices(head_bbs, indices[0]):
        cv2.rectangle(image, (head_bbs[i][0], head_bbs[i][1]), (head_bbs[i][2], head_bbs[i][3]),
                      (0, 0, 255), 2)
        indices_text = '0\t' + str(head_bbs[i][0]) + '\t' + str(head_bbs[i][1]) \
                  + '\t' + str(head_bbs[i][2]) + '\t' + str(head_bbs[i][3])
        text.append(indices_text)
    for i in getMismatchedIndices(person_bbs, indices[1]):
        cv2.rectangle(image, (person_bbs[i][0], person_bbs[i][1]), (person_bbs[i][2], person_bbs[i][3]),
                      (0, 255, 0), 1)
    text = '\t'.join(text) + '\n'

    if len(pair_indices) > 0:
        cover_ratio = cover_ratio / len(pair_indices)
    else:
        cover_ratio = 1.0
    return text, cover_ratio

def getPersonBoundingBoxes(person_dir, filename, swap):
    """
    Read and return a list of person bounding boxes from json detections for an image.
    :param person_dir: directory containing person detections (as json files)
    :param filename: filename of the specific image
    :param swap: True if bounding box coordinates are y1, x1, y2, x2 instead of x1, y1, x2, y2
    :return: list of person bounding boxes
    """
    json_data = json.load(open(os.path.join(person_dir, filename)))
    detections = []
    if 'detections' in json_data.keys():
        detections = json_data['detections']
    person_bbs = [det['bbox'] for det in detections if det['class'] == 'person']
    if swap:
        person_bbs = [[person_bb[1], person_bb[0], person_bb[3], person_bb[2]] for person_bb in person_bbs]
    return person_bbs

def getHeadBoundingBoxes(head_file, person_dir, filename):
    """
     Read and return a list of head bounding boxes from json detections for an image.
    :param head_file: csv file containing annotated head bounding boxes
    :param person_dir: directory containing person detections (as json files)
    :param filename: filename of the specific image
    :return: list of head bounding boxes
    """
    heads = open(head_file, 'r').readlines()
    raw_filename = (person_dir.strip().split('/'))[-1] + '/' + '.'.join((filename.strip().split('.'))[0:-1]) + '.'
    head_line = [line for line in heads if line.find(raw_filename) != -1]
    if len(head_line) == 0:
        return None
    head_bbs = []

    if len(head_line) > 0:  # and len(person_bbs) > 0:
        head_bbs = (head_line[0].strip().split('\t'))[1:]
        head_bbs = [[int(head_bbs[i]), int(head_bbs[i + 1]), int(head_bbs[i + 2]), int(head_bbs[i + 3])] for i
                    in range(len(head_bbs)) if i % 5 == 0]
    return head_bbs

def computeAlginments(head_bbs, person_bbs):
    """
    Compute the head-person matches by solving the assignment problem (Hungarian algorithm)
    :param head_bbs: list of head bounding boxes
    :param person_bbs: list of person bounding boxes
    :return: aligned indices and cost matrix
    """
    indices = np.array([[], []])
    C = np.zeros([len(head_bbs), len(person_bbs)])
    if len(head_bbs) > 0 and len(person_bbs) > 0:
        C = cdist(XA=np.array(head_bbs), XB=np.array(person_bbs), metric=computeIoU)  # maximize
        indices = linear_sum_assignment(C)
    return indices, C

def removeZeroCostIndices(C, aligned_indices):
    """
    Remove indices from matched indices that have 0 cost, since they are not a real match.
    :param C: cost matrix
    :param aligned_indices: paired indices
    :return: indices not containing zero cost pairings
    """
    pair_indices = [(ind1, ind2) for ind1, ind2 in zip(aligned_indices[0], aligned_indices[1])]
    for (row_ind, col_ind) in pair_indices:
        if C[row_ind, col_ind] >= 0:
            (aligned_indices[0].tolist()).remove(row_ind)
            (aligned_indices[1].tolist()).remove(col_ind)
    return aligned_indices

def computeMetrics(C, aligned_indices, head_bbs, person_bbs, cover_ratio, cummulated_metrics):
    """
    Compute metrics for an alignment of an image that is stored in a dictionary
    (cummulated_metrics) that is updated step by step.
    :param C: cost matrix
    :param aligned_indices: paired indices
    :param head_bbs: list of head bounding boxes
    :param person_bbs: list of person bounding boxes
    :param cummulated_metrics: dictionary containing metrices in a cummulated way (not yet finalized)
    :return: updated cummulated_metrics
    """
    aligned_indices = removeZeroCostIndices(C, aligned_indices)
    mismatched_heads = len(getMismatchedIndices(head_bbs, aligned_indices[0]))
    mismatched_people = len(getMismatchedIndices(person_bbs, aligned_indices[1]))
    heads = len(head_bbs)
    people = len(person_bbs)
    cummulated_metrics['count'] += 1
    cummulated_metrics['cover_ratio'] += cover_ratio
    if len(aligned_indices[0]) > 0:
        cost = - C[aligned_indices[0], aligned_indices[1]].sum() / float(len(aligned_indices[0]))
        matched_head_ratio = (heads - mismatched_heads) / heads
        matched_person_ratio = (people - mismatched_people) / people
        matched_objects_ratio = len(aligned_indices[0])/ float(len(aligned_indices[0]) + mismatched_people + mismatched_heads)

        cummulated_metrics['match'] += len(aligned_indices[0])
        cummulated_metrics['cost'] += cost
        cummulated_metrics['matched_head_ratio'] += matched_head_ratio
        cummulated_metrics['matched_person_ratio'] += matched_person_ratio
        cummulated_metrics['matched_object_ratio'] += matched_objects_ratio
    else:
        if heads > 0 and people > 0:
            cummulated_metrics['cost'] += 0.0
            cummulated_metrics['matched_object_ratio'] += 0.0
            cummulated_metrics['matched_head_ratio'] += 0.0
            cummulated_metrics['matched_person_ratio'] += 0.0
        elif heads == 0 and people == 0:
            cummulated_metrics['cost'] += 1.0
            cummulated_metrics['matched_object_ratio'] += 1.0
            cummulated_metrics['matched_head_ratio'] += 1.0
            cummulated_metrics['matched_person_ratio'] += 1.0
        elif heads > 0:
            cummulated_metrics['matched_head_ratio'] += 0.0
            cummulated_metrics['matched_person_ratio'] += 1.0
            cummulated_metrics['cost'] += 0.0
            cummulated_metrics['matched_object_ratio'] += 0.0
        elif people > 0:
            cummulated_metrics['matched_head_ratio'] += 1.0
            cummulated_metrics['matched_person_ratio'] += 0.0
            cummulated_metrics['cost'] += 0.0
            cummulated_metrics['matched_object_ratio'] += 0.0

def finalizeMetrics(cummulated_metrics):
    """
    Compute mean of cummulated metrics
    :param cummulated_metrics: cummulated metrics not yet taken their mean
    :return: metrics with mean computed
    """
    metrics = {'count': 0, 'cost': 0, 'matched_head_ratio': 0.0, 'matched_person_ratio': 0.0,
               'matched_object_ratio': 0.0, 'match': 0.0, 'cover_ratio': 0.0}
    count = cummulated_metrics['count']
    metrics['count'] = count
    metrics['cover_ratio'] = cummulated_metrics['cover_ratio']/float(count)
    metrics['cost'] = cummulated_metrics['cost']/float(count)
    metrics['match'] = cummulated_metrics['match']/float(count)
    metrics['matched_head_ratio'] = cummulated_metrics['matched_head_ratio']/float(count)
    metrics['matched_person_ratio'] = cummulated_metrics['matched_person_ratio']/float(count)
    metrics['matched_object_ratio'] = cummulated_metrics['matched_object_ratio']/float(count)
    return metrics

def Align(head_file, person_dir, image_dir, out_dir, metrics_file, name, swap, reference=None):
    """
    Align heads with people (bodies)
    :param head_file: csv containing information on head bounding boxes
    :param person_dir: directory containing information on person bounding boxes in json files
    :param image_dir: directory containing raw images
    :param out_dir: output directory
    :param metrics_file: file to output gathered metrics
    :param name:
    :param swap: True if person bounding boxes should be treated in a different manner (y1,x1,y2,x2 instead of x1,y1,x2,y2)
    :param reference: directory containing files on which alignments should be run, if None alignments are run on the whole person_dir
    :return:
    """
    cummulated_metrics = {'count': 0, 'cost': 0, 'matched_head_ratio': 0.0, 'matched_person_ratio': 0.0,
                          'matched_object_ratio': 0.0, 'match': 0.0, 'cover_ratio': 0.0}
    if person_dir == '' or image_dir == '': # new version
        suffix = getSuffix(head_file)
        csv_path = suffix + '_head_body.csv'
        print('Output csv ->', csv_path)
        csv_file = open(csv_path, 'w')
        
        heads = open(head_file, 'r').readlines()
        for line in heads:
            parts = line.strip().split('\t')
            file_name = parts[0]
            head_bbs = []
            if len(parts) > 1: 
                head_bbs = parts[1:]
                head_bbs = [[int(head_bbs[i]), int(head_bbs[i + 1]), int(head_bbs[i + 2]), int(head_bbs[i + 3])] for i
                            in range(len(head_bbs)) if i % 5 == 0]

            img_filename = file_name
            json_filename = '.'.join( (img_filename.strip().split('.'))[0:-1] ) + '_head_body.json'
            person_bbs = getPersonBoundingBoxes(person_dir, json_filename, swap)
            if head_bbs is not None:
                indices, C = computeAlginments(head_bbs, person_bbs)
                image = cv2.imread(os.path.join(img_filename))
                csv_text, cover_ratio = drawRectangles(indices, C,  head_bbs, person_bbs, image, img_filename)
                csv_file.write(csv_text)
                #print(img_filename, ' --> ', os.path.join(out_dir, (img_filename+'_aligned.png')))
                #print(img_filename)
                image_file_name_raw = '.'.join((img_filename.split('.'))[0:-1]) + '_aligned.png'
                print(img_filename, ' --> ', os.path.join(out_dir, image_file_name_raw))
                out_path = os.path.join(out_dir, image_file_name_raw)
                #print(out_path)
                if not os.path.exists(os.path.dirname(out_path)):
                    #print("Make dir", os.path.dirname(out_path))
                    os.makedirs(os.path.dirname(out_path))
                cv2.imwrite(out_path, image)
                computeMetrics(C, indices, head_bbs, person_bbs, cover_ratio, cummulated_metrics)
        csv_file.close()
    else: # old version
        file_names = os.listdir(person_dir)
        suffix = getSuffix(person_dir)
        heads = open(head_file, 'r').readlines()
        heads = [(((head.strip().split('/'))[-1]).split('\t'))[0] for head in heads if head.find(suffix) != -1]
        print('heads: ', heads[0:10])
        heads = set(heads)

        img_format = '.png'
        if image_dir.find('HollywoodHeads') != -1:
            img_format = '.jpeg'
        elif image_dir.find('MPII') != -1:
            img_format = '.jpg'
        print('format: ', img_format)
        print('heads: ', len(heads))
        print('suffix: ', suffix)
        file_names = [file_name for file_name in file_names if (file_name.strip().split('.'))[0] + img_format in heads]
        print(file_names[0:10])
        if reference != None:
            reference_names = set(os.listdir(reference))
            file_names = [file_name for file_name in file_names if file_name in reference_names]

        csv_path = suffix + '_head_body.csv'
        print('Output csv ->', csv_path)
        csv_file = open(csv_path, 'w')

        for filename in file_names:
            if filename.find('.json') != -1:
                person_bbs = getPersonBoundingBoxes(person_dir, filename, swap)
                head_bbs = getHeadBoundingBoxes(head_file, person_dir, filename)
                if head_bbs is not None:
                    indices, C = computeAlginments(head_bbs, person_bbs)

                    img_filename = '.'.join((filename.strip().split('.'))[0:-1]) + img_format
                    image = cv2.imread(os.path.join(image_dir, img_filename))
                    csv_text, cover_ratio = drawRectangles(indices, C,  head_bbs, person_bbs, image, os.path.join(person_dir, img_filename))
                    csv_file.write(csv_text)
                    print(img_filename, ' --> ', os.path.join(OUT_DIR, img_filename))
                    cv2.imwrite(os.path.join(OUT_DIR, img_filename), image)
                    computeMetrics(C, indices, head_bbs, person_bbs, cover_ratio, cummulated_metrics)
                    
        csv_file.close()
    metrics = finalizeMetrics(cummulated_metrics)
    metrics['name'] = name
    with open(metrics_file, 'a+') as f:
        f.write('\n')
        json.dump(metrics, f)
    print(metrics)

def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description='(MaskRCNN) body-head aligner')
    parser.add_argument('--head',
                        default='results/head_bounding_boxes/train_v3.csv', type=str,
                        help='Path to annotated head bounding boxes csv file', required=True)
    parser.add_argument('--person', type=str, default='',
                        help='Path to (yolact) directory containing person bounding boxes jsons', required=False)
    parser.add_argument('--images', type=str, default='', 
                        help='Path to directory containing raw images', required=False)
    parser.add_argument('--outdir', default=None, type=str,
                        help='Path to output image directory', required=True)
    parser.add_argument('--metrics', default='metrics.json', type=str,
                        help='Path to output metrics file', required=False)
    parser.add_argument('--name', type=str,
                        help='Name of dataset', required=True)
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to directory containing reference files on which detector should be applied.', required=False)
    parser.add_argument('--swap', type=bool, default=False,
                        help='<True|False>, True if person bounding box coordinates should be swapped', required=False)

    global args
    args = parser.parse_args(argv)

if __name__ == '__main__':
    parseArgs()

    HEAD_FILE = args.head
    PERSON_DIR = args.person
    IMAGE_DIR = args.images
    OUT_DIR = makeOutDir(PERSON_DIR, args.outdir)
    METRICS_FILE = args.metrics
    NAME = args.name
    SWAP = args.swap
    REFERENCE = args.reference
    print("persondir:", PERSON_DIR)
    print("OUT_DIR:",OUT_DIR)
    Align(HEAD_FILE, PERSON_DIR, IMAGE_DIR, OUT_DIR, METRICS_FILE, NAME, SWAP, REFERENCE)
