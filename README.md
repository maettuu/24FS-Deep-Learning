# 24FS Deep Learning (DL)
This repository includes the practical assignments of the course DL. All assignments are written in `Python`.

## Assignment 1: Perceptron Learning
Perception Learning is applied to automatically generated data. The algorithm is initialized with a random line which is then optimized. The results are visualized to show how the learning improves the seperation line.

Includes packages: `numpy`, `matplotlib`

## Assignment 2: Gradient Descent
In this task a two-dimensional loss surface is created manually and gradient descent is implemented including several termination criteria. Several runs of gradient descent from different starting locations are performed. The loss surface and the detected minima are plotted together in one 3D plot.

Includes packages: `numpy`, `matplotlib`, `tqdm`

## Assignment 3: Universal Function Approximator
A two-layer fully-connected network is trained to perform one-dimensional non-linear regression via gradient descent. To show the flexibility of the approach, three different functions will be approximated. First, the network and its gradient are implemented. Second, target data for three different functions is generated. Finally, the training procedure is applied to the data, and the resulting approximated function is plotted together with the data samples.

Includes packages: `numpy`, `math`, `matplotlib`, `tqdm`

## Assignment 4: Multi-Output Networks and Batch Processing
This assignment introduces regularization techniques when implementing deep learning methods. For this purpose, a [dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance) that contains data in different formats, some binary and some numerical and some are categorical is selected. As target values, this dataset contains three numerical outputs. These target values are approximated with a two-layer multi-output network that is trained with the square loss.

Includes packages: `numpy`, `math`, `matplotlib`, `tqdm`, `pandas`

## Assignment 5: Classification in PyTorch
In this exercise some concepts in PyTorch are introduced, such as relying on the `torch.tensor` data structure, implementing the network, the loss functions, the training loop and accuracy computation, which is applied to binary and categorical classification. The [churn dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) is used for binary and the [wine dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) for categorical classification.

Includes packages: `torch`, `matplotlib`, `tqdm`

## Assignment 6: Convolutional Networks
A convolutional network is built, trained and used for prediction applied to the [FashionMNIST dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST). It is compared to a fully-connected network. The results are plotted to compare each network's loss values.

Includes packages: `torch`, `torchvision`, `PIL`, `numpy`, `tqdm`, `matplotlib`

## Assignment 7: Transfer Learning
In this task pre-trained networks are used in transfer learning tasks. Networks trained on ImageNet are used, and applied to related problems, i.e., the classification of categories such as buildings, forests, glaciers, mountains, seas, and streets. The [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) is  used.

Includes packages: `torch`, `torchvision` : {`ImageFolder`}, `numpy`, `tqdm`, `matplotlib`, `sklearn` : {`confusion_matrix`, `ConfusionMatrixDisplay`}

## Assignment 8: Open-Set Classification
In this assignment, a network that is capable of correctly classifying known classes is developed while at the same time rejecting unknown samples that occur during inference time. To showcase the capability, the [MNIST dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST) is used and artificially split into known and unknown classes; this allows to train a network on the data without requiring too expensive hardware.

Includes packages: `torch`, `torchvision`, `tqdm`, `matplotlib`

## Assignment 9: Convolutional Auto-Encoder
This assignment shows that it is possible to learn from unlabeled data using a convolutional auto-encoder network. The task is to reduce an image of the handwritten digits of MNIST into a deep feature representation, without making use of their labels, and reconstruct the sample from that representation. For this purpose, a convolutional auto-encoder is implemented that learns a k-dimensional deep feature representation of each image and uses this representation to reconstruct images to the original size of 28x28 pixels. It is shown that such a network can be used to detect anomalies in the test set. The [FashionMNIST dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST) and [MNIST dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST) are used.

Includes packages: `torch`, `torchvision`, `tqdm`, `matplotlib`, `sklearn` : {`confusion_matrix`}

## Assignment 10: Predicting Stock Prices
A simple recurrent network with a single hidden layer is constructed: a long short-term memory network (LSTM). This network is trained on historical stock market datasets of two distinct companies, and the objective is to leverage them for predicting their future stock prices. The datasets used are: [GAIL stock](https://raw.githubusercontent.com/Pranavd0828/NIFTY50-StockMarket/main/Dataset/GAIL.csv) and [NTPC stock](https://raw.githubusercontent.com/Pranavd0828/NIFTY50-StockMarket/main/Dataset/NTPC.csv).

Includes packages: `torch`, `numpy`, `pandas`, `matplotlib`

## Assignment 11: Adversarial Training
This task shows that adversarial training provides stability against adversarial attacks for the [MNIST dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST). Three different types of training procedures are compared:

- Train with only the original samples
- Train with original samples and samples with added random noise
- Train with original samples and adversarial samples generated by Fast Gradient Sign (FGS) method

Note that the results of this experiment might not translate well to other datasets.

Includes packages: `torch`, `torchvision`, `tqdm`, `matplotlib`

## Assignment 12: Zero Shot Learning and Auto-Labelling Pipeline
The first part of this assignment consists of zero-shot classification (i.e. classifying images into classes without training on them) using the CLIP model on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) containing the following classes: buildings, forests, glaciers, mountains, seas, and streets. A pre-trained VIT model is used for image embeddings. Next, for each class in the dataset a text prompt is created which are then encoded by the CLIP model. Finally, the labels are estimated for each embedding and a confusion matrix shows the accuracy of the results for zero-shot learning via CLIP.

In the second part of this assignment various pre-trained weighted models are used to automatically annotate images (create tags, bounding boxes and segmentation masks) from the [image dataset](https://www.robots.ox.ac.uk/~vgg/data/iseg/data/images.tgz) provided by the University of Oxford VGG group:

- RAM: Recognize Anything Model (Image caption generation)
- GroundingDINO model (Zero-shot detection)
- SAM: Segment Anything Model (Segmentation)

The pipeline then is executed to create the labels, bounding boxes and segmentation masks.

Includes packages: `torch`, `torchvision` : {`ImageFolder`}, `transformers` : {`CLIPProcessor`, `CLIPModel`}, `tqdm`, `sklearn` : {`confusion_matrix`, `ConfusionMatrixDisplay`}, `groundingdino` : {`build_model`, `transforms`, `inference`, `SLConfig`, `utils`}, `segment_anything` : {`build_sam`, `SamPredictor`}, `ram` : {`get_transform`, `inference_ram`, `models.ram`}, `PIL` : {`Image`}, `itertools` : {`chain`}, `cv2`
