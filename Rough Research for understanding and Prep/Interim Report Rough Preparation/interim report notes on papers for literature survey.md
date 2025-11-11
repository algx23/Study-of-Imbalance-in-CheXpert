[Aim]

To design, develop and evaluate a robust CNN architecture to accurately
classify chest x-ray images despite class imbalance being a
characteristic in the dataset

[Objectives]

- Build a Baseline CNN model with 4 convolution layers, and some pooling
  layers

- Evaluate this baseline CNN model with a variety of evaluation metrics,
  such as Accuracy, Recall, Precision and F1 Score

- Add augmentation methods, such as rotations, flipping, colour space
  augmentations

- Evaluate these methods with the same evaluation metrics and compare
  them

- Look into implementing dropout layers, and observe and investigate how
  this changes the performance of the model

[Notes for Interim Report Literature Survey]

[Augmentation / Sampling techniques]

The Effectiveness of Data Augmentation in Image Classification using
Deep Learning:

- Tiny imagenet 200 -- train their CNN to do a rudimentary
  classification, then retrain using traditional augmentation techniques

- They also used CycleGANs to augment the data

- 3 layer network with 2 fully connected layers with dropout

- Took 500 images from cats , and 500 from dogs, and then 500 from
  goldfish rather than cats -- goldfish are less similar to dogs than
  cats, so can kind of see the effects of inter-class similarity --
  important in my project as they're all x-rays

- Ran 10 experiments at 40 epochs each

- For Dogs v Goldfish, using a Nerual Net to augment images and another
  to classify, where the loss is fed back to the first net is the best,
  with an accuracy of 0.915

- For Dogs v Cats, where there is more inter-class similarity,
  traditional methods were better with an accuracy of 0.775 -- suggests
  for my project traditional methods may be better, as all chest x-rays
  are relatively similar regardless of disease class

- Images that perform best have some regularization -- so I should
  probably normalize images again after augmentation (though not sure)

Doi:
[10.1016/j.knosys.2011.06.013](https://doi.org/10.1016/j.knosys.2011.06.013)

- Wanted to find the best method of sampling and investigate the effects
  this had on different levels of class imbalance

- Used 17 real datasets transformed into a 2 class problem -- so One
  class V all other classes

- Found that, for classes with high imbalance, Over sampling is better
  than undersampling for minimising the effect of imbalance

- However, Under sampling is still better than nothing

[Different CNN Architectures]

Medical Image Classification with Convolutional Neural Network
\[https://meichen06.wordpress.com/wp-content/uploads/2017/03/medical-image-classification-with-convolutional-neural-network.pdf\]:

- Solution not problem specific -- easy to apply to different domains

- Lung samples aren't widely available -- so may be good for limited
  data

- Their architecture: convolutional layer with 7x7 filter size and 16
  channels, max pooling layer with a 2x2 filter size, then 100-50-5
  neurons in a fully connected layer, and there is only one
  convolutional layer with ReLu Activation function -- performs as well
  as models with multiple layers in lung patch classification

- Used AVX to optimize CNN -- better than GPGPU based things as it is
  better for running on a general CPU than specialized GPUs

- Only used samples where 75% of pixels fell inside a Annotated Region
  of Interest were used

- CNN got the best performance, and a large margin over SIFT and LBP, so
  automatic feature extraction can be better, though they didn't compare
  with models built specifically for ILD images

[Comparison of 3 different CNN Architectures for Age
Classification]

- Compared a 6 layer CNN and ResNet

- Original image size of 227x227

- 3 convolutional layers, 2 fully connected layers

- ReLu Norm layers after each convolutional layer

- Networks tend to degrade as they get deeper -- so for my project may
  need to think about this

- ResNet is the most successful Residual Network, e.g. 18 convolutional
  layers each followed by ReLu norm in Res-Net 18, no dropout layers
  compared to the plain 6-layer CNN

- Used rotation, visage detection

- Used one-off success (whether the image is classified, or classified
  in one of the neighbours of the correct class) and just normal whether
  the image can be correctly classified

- 6 layer had a mean success of 53.76% compared to ResNet with 52.56%
  and a one-off success of 95.82% compared to ResNet without 92.96% - so
  it suggests that unnecessarily large networks may be more prone to
  overfitting -- must think about this in my project

\####### start here when continuing to write the lit review#####

Tackling Inter-Class Similarity and Intra Class Variance in microscopic
image classification \[https://arxiv.org/pdf/2109.11891\]:

- Divised an algorithm to deal with inter class similarity and intra
  class variance

- Inter class similarity -- when images of two different classes look
  similar

- Intra class variance -- when two images of the same class have
  dramatically different appearances

- Proposed a triple loss function to separate feature embeddings between
  classes, and it tries to minimise the distance between objects
  belonging to the same class, while maximising the distance between
  different classes

- For intra class variance they suggested using x means clustering - but
  too many clusters doesn't solve the problem so they want to find the
  optimum number of clusters that minimize both inter class similarity
  and intra class variance -- X means has a parameter for the upper
  limit of the number of clusters and they say their solution decides
  this to find the optimum number

- Found that they both perform better than standard classification, in
  terms of F1 Score, but then they also found that clustering performs
  better than triplet loss -- so I could think about this for my project
  too, as it may be easier to implement too potentially (guessing)

- Can also use BOTH of them together

CTrans CNN -- Combining Transformer and CNN in medical Image
classification
\[https://www.sciencedirect.com/science/article/pii/S0950705123007803\]:

- deerThey say that if there is a class that is really rare, the
  performance on that class with standard CNNs is not that good due to
  limits in how the kernels can represent the image

- They say a limitation is in some conditions there may be
  interdependency between anatomical structures, so like cardiac
  anomalies in CXR, and also getting enough data is difficult

- Introduced label embeddings to help with the first problem and cross
  attention for the second problem -- label embeddings are represented
  as dense vectors in a continuous space so you can see similarities
  between labels better, and cross attention is

- Method is to: have a transformer and a cnn -- extract features from
  the images, and send copies to the transformer branch and the CNN
  branch. Then the transformer adopts label embedding, and the CNN
  branch stacks MMAEF and MBR

- Then after obtaining the features, investigate fusion methods to
  classify them

- On Chest X Ray 14, the CTrans CNN had the highest mean AUC score on 7
  of the 11 categories

- They also say that ResNet 50 outperformed ResNet 34 so it kind of
  contradicts the Age classification study that said they get worse as
  they get deeper -- though this is not HUGELY deepr so it could be that

Different Kernel Size Impact on CNN Classification accuracy
\[https://ieeexplore.ieee.org/abstract/document/9142089\]:

- Lower layers use larger filters, and higher layers use smaller filters

- Prepared data with normalization and noise removal -- but didn't
  augment so on augmented images, it may be different?

- They used an image with 50x50 pixels

- Convolution layer 1 had 32 3x3 filters, then a pooling layer with a
  2x2 size and stride 2,a hidden layer with 128 nodes, and 2 nodes as
  output

- In other experiments, the number of filters in the layers are 64 and
  128, with 3x3 filters

- In experiemt 4 they use 32, 64 and 128 filters with a size of
  5x5pixels,

- In experiment 5, the structure is the same as 4, but the filter size
  is 7x7

- Then they redid the experiment with 4 convolution layers and 2 pooling
  layers, rather than 2 convolution layers and 1 pooling layer

- They found that with 3x3 filters, 32 filters had higher accuracy than
  64, but 128filters with one convolution layer had a accuracy of 98.47%

- With 2 convolution layers, and a 7x7 size it had a lower accuracy with
  32 filters (96.98%) compared to the same architecture with 32 3x3
  filters (97.84) -- suggests more filters with smaller filter
  dimensions could be a good balance

- Training time increased as the size of the filter increased, but 32
  filters always resulted in the lowest training times

Feature Extraction and Classification of Chest X Ray Images Using CNN to
detect Pneumonia
\[https://ieeexplore.ieee.org/abstract/document/9057809\]:

- Wanted to build a CNN from scratch, as they wanted to be able to
  classify images as pneumonia or not, and models that can be used to
  use transfer learning have millions of parameters, and are taking a
  lot of time and resources to train

- Two CNN architectures used -- One with a dropout layer, and one
  without a dropout layer

- First part has 2 convolution layers, with 32-32 units, with a max
  pooling layer of size 3x3 and a ReLu Activator, the other has 2
  convolution layers, but with 64, and 128 filters with a max pooling
  layer of size 2x2 and ReLu

- ReLu introduces non linearity to the model, which is good because it
  allows the model to learn more complex features

- Then the output from this is passed to a flatten layer to give a 1
  dimensional output which is fed to the dense layer

- Dropout -- neurons have a probability p of being dropped -- so all
  inputs and outputs from the neuron are deactivated, so there is some
  loss of data but it can enhance regularization so the model can
  generalize better and predict with higher accuracy, a dropout of 0.5
  means 50% of neurons are dropped

- Without augmentation, with dropout had a lower loss on training data
  than with augmentation with dropout,

- On validation accuracy, with augmentation, with dropout had a lower
  accuracy than without augmentation, with dropout until 18 epochs, so
  maybe at a lower number of epochs if constrained, augmentation makes
  less of a difference than dropout does

- Overall testing accuracy higher with dropout and augmentation
  (0.9068), so I can use both perhaps, especially if I have a lower
  number of epochs on a limited system

[Requirement Analysis]

[Functional]

- The model must normalize and augment images in a way that maximises
  performance within the constraints

- The user should be able to drag and drop an unseen image into the
  program, which will then be classified

- The system should be able to handle PNGs and (MAYBE JPEGS)

- The system should show the class label it predicted, and the
  confidence, so that it can be checked over, and is not seen as fact in
  medical imaging and can be rechecked over with experts

[Non Functional]

- New unseen images should be able to be classified within 20seconds

- The system should ensure that the original images, are not altered
  themselves, and that any augmentations are completed on a copy of the
  image
