from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
# from cards_class import showBatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'data/models/cards.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

rev_label_map = labelMap('data/general_labels/classes.txt')
print('Classes:\n', rev_label_map)
label_map = invLabelMap(rev_label_map)


distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', 
                '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', 
                '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', 
                '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080', 
                '#FFFFFF', '#c6e2ff', '#b0e0e6', '#666a67', '#4beddc', 
                '#ae415d', '#c1d8c3', '#ffc0cb', '#fb6781', '#f42069',
                '#ffb6c1', '#102099', '#7c4780', '#a0cf8d', '#f42069',
                '#fb6781', '#89cff0', '#ffb6c1', '#f42069', '#000000',
                '#d9534f', '#4a95ff', '#debc97', '#215b8f', '#6fb407', 
                '#5f8c4a', '#c6ac54', '#8d5d75', '#280050', '#508ef8',
                '#d109e4', '#534275', '#2761ee', '#15a5ff', '#bd0bad',
                '#9e0ea1']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}



def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

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
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]


    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        w, h = original_image.size
        if h > 500:
            fact = 500 / h
            fact_w = int(fact * w)
            fact_h = int(fact * h)
            original_image = transforms.Resize((fact_h, fact_w))(original_image)
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
    w, h = annotated_image.size
    if h > 500:
        fact = 500 / h
        fact_w = int(fact * w)
        fact_h = int(fact * h)
        annotated_image = transforms.Resize((fact_h, fact_w))(annotated_image)

    del draw

    return annotated_image


if __name__ == '__main__':
    img_path = 'data/images/Card_492.jpeg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    pred_img = detect(original_image, min_score=0.1, max_overlap=.5, top_k=5).show()
    