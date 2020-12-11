from cards_class import CardsDataset
from utils import unpackBoundigBox, show, collate_fn, calculate_mAP, cxcy_to_xy
from tqdm import tqdm
from pprint import PrettyPrinter
from torch.utils.data import DataLoader
import torch

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
# data_folder = './'
batch_size = 2
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './data/models/cards.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()



# Load test data
images_path = 'data/images'
anotations_path = 'data/txt_cards'
labels_path = 'data/general_labels/classes.txt'
# wrong it has to be different dataset
test_dataset = CardsDataset(images_path, anotations_path, labels_path)
all_labels = test_dataset.invLabels

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn, drop_last=True)


def evaluate(test_loader, model, all_labels):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            difficulties = [torch.zeros(l.size()) for l in labels]
            boxes = [cxcy_to_xy(b).to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)



        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, all_labels)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model, all_labels)