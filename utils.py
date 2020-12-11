import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
from torchvision import transforms
import time
from PIL import Image, ImageDraw, ExifTags
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def labelMap(labels_path):
    labels_path = os.path.join(labels_path)
    labels = dict()
    labels[0] = 'background'
    file = open(labels_path, 'r')
    idx = 1 
    for line in file:
        # if idx < 14:
        if line.endswith('\n'):
            labels[idx] = line[:-1]
        else:
            labels[idx] = line
        idx += 1
    file.close()
    return labels

def invLabelMap(labels):
    inv_labels = {labels[key]: idx for idx, key in enumerate(list(labels.keys()))}
    return inv_labels

def readAnotationTxt(anotation_path):
    anotation_file = open(anotation_path, 'r')
    boxes = list()
    labels = list()
    for line in anotation_file:
        label_id, x, y, w, h = line.strip(' \n').split()
        labels.append(int(label_id))
        boxes.append([float(x), float(y), float(w), float(h)])
    return labels, boxes

def unpackBoundigBox(bbox, img):
    if isinstance(img, str):
        img_w, img_h = Image.open(img, mode='r').size
    else:
        img_w, img_h = img.size
    scaled_box = list()
    if bbox is not None and isinstance(bbox, torch.Tensor):
        if isinstance(bbox[0], torch.Tensor):
            for box in bbox:
                x, y, w, h = box
                x_min = int((x - w/2) * img_w)
                x_max = int((x + w/2) * img_w)
                y_min = int((y - h/2) * img_h)
                y_max = int((y + h/2) * img_h)
                scaled_box.append([x_min, y_min, x_max, y_max])
        else:
            x, y, h, w = bbox
            x_min = int((x - w/2) * img_w)
            x_max = int((x + w/2) * img_w)
            y_min = int((y - h/2) * img_h)
            y_max = int((y + h/2) * img_h)
            scaled_box.append([x_min, y_min, x_max, y_max])
    return scaled_box

def showBatch(batch, factor=1, pred_bbox=None):
    img, bbox, labels = batch

    img = transforms.ToPILImage()(img)
    w, h = img.size
    fact_w = int(factor * w)
    fact_h = int(factor * h)
    img = transforms.Resize((fact_h, fact_w))(img)
    
    draw = ImageDraw.Draw(img)

    bbox = unpackBoundigBox(bbox, img)
    for rect in bbox:
        draw.rectangle(rect)
    img.show()

def show(image, factor=1):
    w, h = image.size
    fact_w = int(factor * w)
    fact_h = int(factor * h)
    img = transforms.Resize((fact_h, fact_w))(image)
    img.show()

def collate_fn(batch):
    images = list()
    boxes = list()
    labels = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])

    images = torch.stack(images, dim=0)
    return images, boxes, labels

def train(train_loader, model, criterion, optimizer, epoch, print_freq=50, grad_clip=None):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):
        # (images, boxes, labels) = next(iter(train_loader))
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [cxcy_to_xy(b).to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

def detect(model, rev_label_map, original_image, min_score, max_overlap, top_k, device, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0',
                '#f032e6', '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8',
                '#800000', '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080',  '#FFFFFF',
                '#c6e2ff', '#b0e0e6', '#666a67', '#4beddc', '#ae415d', '#c1d8c3', '#ffc0cb',
                '#fb6781', '#f42069', '#ffb6c1', '#102099', '#7c4780', '#a0cf8d', '#f42069',
                '#fb6781', '#89cff0', '#ffb6c1', '#f42069', '#000000', '#d9534f', '#4a95ff',
                '#debc97', '#215b8f', '#6fb407', '#5f8c4a', '#c6ac54', '#8d5d75', '#280050',
                '#508ef8', '#d109e4', '#534275', '#2761ee', '#15a5ff', '#bd0bad', '#9e0ea1']
    
    label_map = invLabelMap(rev_label_map)
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
    
    

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    # Transform
    image = transform(original_image)

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, 
                                                            predicted_scores, 
                                                            min_score=min_score,
                                                            max_overlap=max_overlap,
                                                            top_k=top_k)


    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    img_dim = [original_image.width, original_image.height, original_image.width, original_image.height]
    original_dims = torch.FloatTensor(img_dim).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]


    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/Nakula/nakula.ttf", 100)
    except OSError:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 100)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                             box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image



def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.
    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [voc07_path, voc12_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Test data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in the test data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, labels):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation
    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(labels)
    rev_label_map = invLabelMap(labels)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    In the model, we are predicting bounding box coordinates in this encoded form.
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
    They are decoded into center-size coordinates.
    This is the inverse of the function above.
    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

# def expand(image, boxes, filler):
#     """
#     Perform a zooming out operation by placing the image in a larger canvas of filler material.
#     Helps to learn to detect smaller objects.
#     :param image: image, a tensor of dimensions (3, original_h, original_w)
#     :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
#     :param filler: RBG values of the filler material, a list like [R, G, B]
#     :return: expanded image, updated bounding box coordinates
#     """
#     # Calculate dimensions of proposed expanded (zoomed-out) image
#     original_h = image.size(1)
#     original_w = image.size(2)
#     max_scale = 4
#     scale = random.uniform(1, max_scale)
#     new_h = int(scale * original_h)
#     new_w = int(scale * original_w)

#     # Create such an image with the filler
#     filler = torch.FloatTensor(filler)  # (3)
#     new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
#     # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
#     # because all expanded values will share the same memory, so changing one pixel will change all

#     # Place the original image at random coordinates in this new image (origin at top-left of image)
#     left = random.randint(0, new_w - original_w)
#     right = left + original_w
#     top = random.randint(0, new_h - original_h)
#     bottom = top + original_h
#     new_image[:, top:bottom, left:right] = image

#     # Adjust bounding boxes' coordinates accordingly
#     new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
#         0)  # (n_objects, 4), n_objects is the no. of objects in this image

#     return new_image, new_boxes


# def random_crop(image, boxes, labels, difficulties):
#     """
#     Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.
#     Note that some objects may be cut out entirely.
#     Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
#     :param image: image, a tensor of dimensions (3, original_h, original_w)
#     :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
#     :param labels: labels of objects, a tensor of dimensions (n_objects)
#     :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
#     :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
#     """
#     original_h = image.size(1)
#     original_w = image.size(2)
#     # Keep choosing a minimum overlap until a successful crop is made
#     while True:
#         # Randomly draw the value for minimum overlap
#         min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

#         # If not cropping
#         if min_overlap is None:
#             return image, boxes, labels, difficulties

#         # Try up to 50 times for this choice of minimum overlap
#         # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
#         max_trials = 50
#         for _ in range(max_trials):
#             # Crop dimensions must be in [0.3, 1] of original dimensions
#             # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
#             min_scale = 0.3
#             scale_h = random.uniform(min_scale, 1)
#             scale_w = random.uniform(min_scale, 1)
#             new_h = int(scale_h * original_h)
#             new_w = int(scale_w * original_w)

#             # Aspect ratio has to be in [0.5, 2]
#             aspect_ratio = new_h / new_w
#             if not 0.5 < aspect_ratio < 2:
#                 continue

#             # Crop coordinates (origin at top-left of image)
#             left = random.randint(0, original_w - new_w)
#             right = left + new_w
#             top = random.randint(0, original_h - new_h)
#             bottom = top + new_h
#             crop = torch.FloatTensor([left, top, right, bottom])  # (4)

#             # Calculate Jaccard overlap between the crop and the bounding boxes
#             overlap = find_jaccard_overlap(crop.unsqueeze(0),
#                                            boxes)  # (1, n_objects), n_objects is the no. of objects in this image
#             overlap = overlap.squeeze(0)  # (n_objects)

#             # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
#             if overlap.max().item() < min_overlap:
#                 continue

#             # Crop image
#             new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

#             # Find centers of original bounding boxes
#             bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

#             # Find bounding boxes whose centers are in the crop
#             centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
#                     bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

#             # If not a single bounding box has its center in the crop, try again
#             if not centers_in_crop.any():
#                 continue

#             # Discard bounding boxes that don't meet this criterion
#             new_boxes = boxes[centers_in_crop, :]
#             new_labels = labels[centers_in_crop]
#             new_difficulties = difficulties[centers_in_crop]

#             # Calculate bounding boxes' new coordinates in the crop
#             new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
#             new_boxes[:, :2] -= crop[:2]
#             new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
#             new_boxes[:, 2:] -= crop[:2]

#             return new_image, new_boxes, new_labels, new_difficulties


# def flip(image, boxes):
#     """
#     Flip image horizontally.
#     :param image: image, a PIL Image
#     :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
#     :return: flipped image, updated bounding box coordinates
#     """
#     # Flip image
#     new_image = FT.hflip(image)

#     # Flip boxes
#     new_boxes = boxes
#     new_boxes[:, 0] = image.width - boxes[:, 0] - 1
#     new_boxes[:, 2] = image.width - boxes[:, 2] - 1
#     new_boxes = new_boxes[:, [2, 1, 0, 3]]

#     return new_image, new_boxes


# def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
#     """
#     Resize image. For the SSD300, resize to (300, 300).
#     Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
#     you may choose to retain them.
#     :param image: image, a PIL Image
#     :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
#     :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
#     """
#     # Resize image
#     new_image = FT.resize(image, dims)

#     # Resize bounding boxes
#     old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
#     new_boxes = boxes / old_dims  # percent coordinates

#     if not return_percent_coords:
#         new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
#         new_boxes = new_boxes * new_dims

#     return new_image, new_boxes


# def photometric_distort(image):
#     """
#     Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
#     :param image: image, a PIL Image
#     :return: distorted image
#     """
#     new_image = image

#     distortions = [FT.adjust_brightness,
#                    FT.adjust_contrast,
#                    FT.adjust_saturation,
#                    FT.adjust_hue]

#     random.shuffle(distortions)

#     for d in distortions:
#         if random.random() < 0.5:
#             if d.__name__ is 'adjust_hue':
#                 # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
#                 adjust_factor = random.uniform(-18 / 255., 18 / 255.)
#             else:
#                 # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
#                 adjust_factor = random.uniform(0.5, 1.5)

#             # Apply this distortion
#             new_image = d(new_image, adjust_factor)

#     return new_image


# def transform(image, boxes, labels, difficulties, split):
#     """
#     Apply the transformations above.
#     :param image: image, a PIL Image
#     :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
#     :param labels: labels of objects, a tensor of dimensions (n_objects)
#     :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
#     :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
#     :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
#     """
#     assert split in {'TRAIN', 'TEST'}

#     # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
#     # see: https://pytorch.org/docs/stable/torchvision/models.html
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]

#     new_image = image
#     new_boxes = boxes
#     new_labels = labels
#     new_difficulties = difficulties
#     # Skip the following operations for evaluation/testing
#     if split == 'TRAIN':
#         # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
#         new_image = photometric_distort(new_image)

#         # Convert PIL image to Torch tensor
#         new_image = FT.to_tensor(new_image)

#         # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
#         # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
#         if random.random() < 0.5:
#             new_image, new_boxes = expand(new_image, boxes, filler=mean)

#         # Randomly crop image (zoom in)
#         new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
#                                                                          new_difficulties)

#         # Convert Torch tensor to PIL image
#         new_image = FT.to_pil_image(new_image)

#         # Flip image with a 50% chance
#         if random.random() < 0.5:
#             new_image, new_boxes = flip(new_image, new_boxes)

#     # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
#     new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

#     # Convert PIL image to Torch tensor
#     new_image = FT.to_tensor(new_image)

#     # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
#     new_image = FT.normalize(new_image, mean=mean, std=std)

#     return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(filename, epoch, model, optimizer):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)