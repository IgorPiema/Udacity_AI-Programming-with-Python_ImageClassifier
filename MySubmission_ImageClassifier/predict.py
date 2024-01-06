# -*- coding: utf-8 -*-
# */Artificial Intelligence Programming with Python - Create Your Own Image Classifier/*
#                                                                             
# PROGRAMMER: Gildas Igor Piema-Fangonda
# DATE CREATED: 01/12/2023                                
# REVISED DATE: 08/12/2023

# PURPOSE: Predict flower name from an image with predict.py along with the probability of that name. 
#          That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
#
#		   Basic usage - python predict.py /path/to/image checkpoint
#
#		   Prints out training loss, validation loss, and validation accuracy as the network trains
#
#		   Option -  *Return top KK most likely classes: python predict.py input checkpoint --top_k 3 
#
#					 *Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
#
#                    *Use GPU for inference: python predict.py input checkpoint --gpu 
#
#
##

# This imports the libraries

#load libraries
import argparse
import json
import modeler
import data_manager

parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path', help='Path to image')
parser.add_argument('checkpoint', help='Given checkpoint of a network')
parser.add_argument('--top_k', help='Return top k most likely classes')
parser.add_argument('--category_names', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', help='Use GPU for inference', action='store_true')

args = parser.parse_args()

top_k = 1 if args.top_k is None else int(args.top_k)
category_names = "cat_to_name.json" if args.category_names is None else args.category_names
gpu = False if args.gpu is None else True

model = modeler.load_model(args.checkpoint)
print(model)

probs, predict_classes = modeler.predict(data_manager.process_image(args.image_path), model, top_k)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

classes = []
    
for predict_class in predict_classes:
    classes.append(cat_to_name[predict_class])

print(probs)
print(classes)

