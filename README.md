# Playing Cards

This repository implements Single Shot MultiBox Detector (SSD) on a costume dataset. The objective is to detect playing cards' club and number.

The SSD network is implemented with Pytorch library.

### Files
The files in the repository are 
- ```cards_class.py```: Dataset class
- ```detect.ipynb / detect.py```: Use a model to dtect objects in a given image
- ```eval.ipynb / eval.py```: Evaluate the model using *Mean Average Precision (mAP)*
- ```model.py```: SSD Neural Network using VGG-16 architecture and pretrained model for clasification 
- ```relabeling.py```: Relabeling if more images are added 
- ```split_labels.py```: Remove not wanted labels
- ```train.ipynb / train.py```: Train the SSD Neural Network with costum dataset
- ```utils.py```: Functions used by other files

### Dataset
For the dataset, standard Bicicle Playing Cards where used for training and validation sets. Images where taken from different lightning, rotation and size conditions.

The images are stored in ```data/images```, their anotations are in ```data/txt_cards``` and the labels are in ```data/general_labels```.

### Model
The model's output is stored in ```data/models```.



### References

I used as a reference *sgrvind's* github repository  [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) in which the concepts behind this neural network is very well explained.

