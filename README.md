## **Udacity-AI-programming-with-python_ImageClassifier**    
The project is the second project of [AI Programming with Python nanodegree program](https://www.udacity.com/enrollment/nd089/8.0.17) offered by [Udacity](https://www.udacity.com/enrollment/nd089/8.0.17). The tasks consist in building and training a deep neural network on flowers dataset then convert it into an application that others can use in two parts.

**Part 1**    
In the part 1, in a Jupyter notebook, the main tasks consist in:

  * reading and transforming the flowers dataset
  * choosing the right network architecture (vgg13, vgg16, alexnet)
  * defining a suitable image classifier corresponding to the architecture
  * training a deep neural network
  * saving and loading checkpoints of the neural network
  * illustrating the predictions with the corresponding probability


**Part 2**    
In part 2, the built and trained neural network is converted into a command line application that different users can use to:

  * make a choice of different network architecture
  * customize the hyperparameters (epochs, learning_rate, hidden units)
  * use a GPU for a fast training when available
  * save and load the model
  * use the model to make predictions
    
## **Training**
  * in the repository get access to the directory **ImageClasifier/**
  * in the directory above run the command **python.py <path_to_image> <checkpoint>**
  * to have all the customizations run the following command **python train.py --h**

## **Predictions**
  * in the repository get access to the directory **ImageClassifier/**
  * in the directory run the command **predict.py <path_to_image> <checkpoint>**
  * to have all the customizations run the following command **python predict.py --h**
