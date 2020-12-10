import pandas as pd
import os
import glob
import sys
import cv2


def getClass(path):
    file = open(path, 'r')
    class_dict = dict()
    idx = 1

    for line in file:
        if line.endswith('\n'):
            class_dict[line[:-1].strip()] = idx
        else:
            class_dict[line.strip()] = idx
        idx += 1

    return class_dict


def invDict(data):
    return {data[key]: key for key in list(data.keys())}


def relabelTxtFiles(files_path, actual_class, last_class):
    inv_last_class = invDict(last_class)
    print(inv_last_class)
    files = list(sorted(glob.glob(files_path)))
    for file in files:
        f = open(file, 'r')
        txt = ''
        for line in f:
            label_id, x, y, w, h = line.strip(' \n').split()
            txt += ' '.join([str(actual_class[inv_last_class[int(label_id) + 1]]), x, y, w, h, '\n'])
        f.close()
        print(file)
        # print(txt)


        f = open(file, 'w')
        f.write(txt)
        f.close()


def main():
    # labels = ['a', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'spade' ,'club', 'heart', 'diamond', 'joker']
    # cards = Relabeling(labels)

    # class_id = cards.labels2DataFrame
    # class_id.to_csv(path_or_buf='data/general_labels/classes.csv')

    actual_class_path = 'data/general_labels/classes.txt'
    last_class_path = 'data/classes_2.txt'

    classes = getClass(actual_class_path)
    last_classes = getClass(last_class_path)

    print(classes)
    # print(last_classes)

    txt_path = 'data/txt_cards_2/Card*.txt'
    # relabelTxtFiles(txt_path, classes, last_classes)



    

if __name__ == '__main__':
    main()