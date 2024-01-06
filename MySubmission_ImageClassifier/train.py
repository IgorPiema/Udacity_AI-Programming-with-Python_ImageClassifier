# */Artificial Intelligence Programming with Python - Create Your Own Image Classifier/*
#                                                                             
# PROGRAMMER: Gildas Igor Piema-Fangonda
# DATE CREATED: 01/12/2023                                
# REVISED DATE: 08/12/2023

# PURPOSE: Train a network on a new data set with train.py
#
#		   Basic usage - python train.py data_directory
#
#		   Prints out training loss, validation loss, and validation accuracy as the network trains
#		   Option -  *Set directory to save checkpoints: python train.py data_dir--save_dir-- save_directory
#					 *Choose architecture: python train.py data_dir -- arch "vgg13"
#                    *set hyperparameters: python train.py data_dir -- learning_rate 0.01 --hidden units 512 -- epochs 20 
#                    *Use GPU for training: python train.py --gpu


#  Create a function that retrieves the following 3 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##
# Importing libraries
import argparse
from data_manager import load_data
import modeler

parser = argparse.ArgumentParser(description='Training a neural network on a given dataset')
parser.add_argument('data_directory', help='Path to dataset on which the neural network should be trained on')
parser.add_argument('--save_dir', help='Path to directory where the checkpoint should be saved')
parser.add_argument('--arch', help='Network architecture (default \'vgg16\')')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='Number of hidden units')
parser.add_argument('--epochs', help='Number of epochs')
parser.add_argument('--gpu', help='Use GPU for training', action='store_true')


args = parser.parse_args()


save_dir = '' if args.save_dir is None else args.save_dir
network_architecture = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.0025 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True


train_data, trainloader, validloader, testloader = load_data(args.data_directory)


model = modeler.build_network(network_architecture, hidden_units)
model.class_to_idx = train_data.class_to_idx

model, criterion = modeler.train_network(model, epochs, learning_rate, trainloader, validloader, gpu)
modeler.evaluate_model(model, testloader, criterion, gpu)
modeler.save_model(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)

