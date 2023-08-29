# MLProjectFall21

Our project tackles the problem of the COVID-19 respiratory disease. For this reason, the project aims to provide an alternate solution through proposing a deep learning-based model that diagnoses COVID using chest x-ray images. In this project, several experiments are conducted involving pretraining through either transfer learning (TL) and self-supervised learning (SSL), to then fine-tune for the downstream task at hand. For the project, baseline experiments are performed using ResNet-18 and DenseNet-121, separately. Moreover, the project TL experiments with TL include pretraining with ImageNet, once using ResNet-18 and once DenseNet-121, as well as pretraining with CheXpert, again, once with ResNet-18 and once using DenseNet-121. On the other hand, the SSL experiments involve pretraining with CheXpert and using ResNet-18 only. This however, is done using two different SSL approaches which are SimCLR and MoCo. 


1. Download the dataset from https://www.kaggle.com/c/siim-covid19-detection.
There is a folder named RSNACOVID which you will need to unzip. Inside this folder are a test folder, train folder, and csv files. We only use the train_study_level.csv which includes a one-hot vector for every image, identifying which class it belongs to. Kindly make sure that the folder has the dataset is called RSNACOVID and is in the same directory as the code. 
 
 
2. Download the CheXpert images from https://www.kaggle.com/mimsadiislam/chexpert. Kindly make sure that the folder containing the images is named CheXpert-v1.0-small images. This folder contains a subset of cheXpert. Also, ensure that the folder is in the same directory as the code.


3. In order to preprocess data, run the preprocessing.py file. This is expected to run once to preprocess data and not used again.


4. Run the main code through the script file specifications.sh. This file includes a number of input specifications which are:
   1. The experiment (technique) you wish to conduct: baseline model (ResNet-18 or DenseNet-121), transfer learning (ImageNet or CheXpert) (ResNet-18 or DenseNet-121), or self-supervised learning (SimCLR or MoCo)
   2. Several paths required for multiple parts of the code
   3. Batch size for the experiment
   4. Number of epochs
   5. Whether you wish to show a histogram for the data (default is false)
   6. Whether you wish to show the transformed image (default is false)
   7. Whether you want to show the key regions in a specific image as highlighted by the GradCam (default is false)


5. The main code would run your desired experiments with the help of the other '.py' files. The job for each is as follows:
   1. data.py: includes the class to define a chest x-ray dataset as well functions to show histogram for data classes, show transformed images, calculate class weights, and perform mean and standard deviation calculations
   2. train.py: includes a training class which is the base for the training of the different experiments
   3. test.py: includes a testing class which is the base for testing of the different experiments
   4. baseline.py: includes functions for baseline (ResNet-18) and baseline (DenseNet-121)
   5. transfer.py: includes functions for transfer learning (ImageNet or CheXpert) (ResNet-18 or DenseNet-121)
   6. preprocessing_train_chexpert.py: includes a data set preparation function as well as a preprocessing/training function that is essential for the transfer learning CheXpert-pretrained experiments
   7. SimCLR_pretrain.py: builds SimCLR model and saves the model
   8. SimCLR.py: loads model from SimCLR_pretrain.py to carry on SimCLR downstream experiment
   9. MoCo_pretrain.py: builds MoCo model and saves the model
   10. MoCo.py: loads model from MoCo_pretrain.py to carry on MoCo downstream experiment
   11. GradCam.py: includes function to highlight key regions in the image for a particular prediction
   12. Analysis.py: includes functions essential for model evaluation such as a confusion matrix function and learning curve plotting function


Note: The repository contains a demo file in which one of the experiments is conducted. The demonstrated experiment involves the case of transfer learning, pretained on ImageNet, using ResNet-18.
