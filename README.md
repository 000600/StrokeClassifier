# Stroke Classifier

## The Neural Network

This neural network predicts whether or not a patient will have a stroke based on multiple different factors. The model will predict a value close to 0 if the patient is predicted to be fine (not have a stroke) and a 1 if the patient is predicted to have a stroke. Since the model only predicts binary categorical values, the model uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001 and multiple dropout layers to prevent overfitting. The model has an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 128 input neurons and a ReLU activation function)
- 2 Hidden layers (each with 64 neurons and a ReLU activation function)
- 3 Dropout layers (one after each hidden layer and input layer and each with a dropout rate of 0.4)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset. Credit for the dataset collection goes to **whxna-0615**, **stpete_ishii**, and others on *Kaggle*. It describes whether or not a person will have a stroke (encoded as 0 or 1) based on multiple factors, including:
- Age
- Hypertension (0 : no patient hypertension, 1 : hypertension within patient)
- Average glucose level 
- Body mass index (BMI)
- Smoking status

Note that the initial dataset is biased (this statistic can be found on the data's webpage); it contains a higher representation of non-stroke cases (encoded as 0's in this model) than stroke cases (encoded as 1's in this model). This issue is addressed within the classifier file using Imbalanced-Learn's **SMOTE()**, which oversamples the minority class within the dataset.

## Libraries
This neural network was created with the help of the Tensorflow, Imbalanced-Learn, and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
- Imbalanced-Learn's Website: https://imbalanced-learn.org/stable/about.html
- Imbalanced-Learn's Installation Instructions: https://pypi.org/project/imbalanced-learn/

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
