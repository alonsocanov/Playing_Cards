from utils import show, labelMap, detect
from PIL import Image
import torch


def main():
    # from cards_class import showBatch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = 'data/models/cards.pth.tar'
    labels_path = 'data/general_labels/classes.txt'

    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    model = checkpoint['model']
    model = model.to(device)
    model.eval()


    rev_label_map = labelMap(labels_path)


    print('Classes:\n', rev_label_map)
    print('Loaded checkpoint from epoch', start_epoch)

    img_path = 'data/images/Card_492.jpeg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    pred_img = detect(model=model, rev_label_map=rev_label_map, original_image=original_image, 
                    min_score=0.1, max_overlap=.5, top_k=5, device=device)
    show(pred_img, factor=.1)


if __name__ == '__main__':
    main()
    