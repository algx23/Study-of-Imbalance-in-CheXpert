Classification of medical X Ray Images

Motivation of this project

The NHS have said that 84-94% of image diagnostics are completed within
2 weeks. I believe that with this project, if improved upon in the
future, would be able to cut down the time taken to diagnose based on
images, such as x ray images, to a matter of days.

What is Classification

Classification is an aspect of Machine Learning, which uses statistical
methods to acquire knowledge about a task and improve its performance,
without needing to be directly programmed to do so (Alnuaimi and
Albaldawi, 2024)[^1]. More specifically, classification is a type of
Supervised Learning, which means that the system uses already labelled
data to correctly be able to predict outcomes -- for example, in the
context of medical x-ray images, whether a given X ray image of a
humerus (the bone from the shoulder to the elbow), is fractured, or
normal.

Classification is a method in which data is categorized into distinct
classes. It groups objects, such as medical x ray images, into
pre-existing categories/labels. The overall aim of classification is to
take an unknown data point -- such as an x ray image, and assign it to a
known category -- such as "broken" or "not broken", based on certain
features present in the image. There are many different types of
classification algorithms, which include: Ensemble Algorithms, K-Nearest
Neighbors, Decision Trees and Logistical Regression.

Why use classification on images

Classification of images, particularly medical x ray images, can be a
crucial step forward in regards to being able to accurately support
medical staff when diagnosing certain conditions. One of the key
advantages of using automatic classification on medical x ray images,
especially, is that it may be able to diagnose conditions much faster
and more accurately than humans. For example, a study by Yilun Wang and
Michal Kosinski found that Deep Neural Networks were more accurate than
humans at detecting the sexual orientation of a human, from pictures of
their face (Wang and Kosinski, 2018)[^2]. This suggests that, in the
future, the use of neural networks in classifying images of X Rays may
have the potential to surpass human abilities, leading to more accurate
diagnosis. This can ultimately lead to greater survival rates for many
patients, owing to them being diagnosed accurately, much quicker than if
a human doctor had worked purely by themselves when diagnosing the
patient.

While, due to privacy, and ethical concerns, hospitals likely would not
exclusively use Machine Learning models to classify X Ray images, the
research seems to suggest the potential is real, to have them aid in
diagnosing certain conditions.

The Different Phases of Classification:

The training phase

The training phase of classification is when the model learns from the
data to for example, be able to accurately classify, or categorize,
whether an x ray image of a humerus has a fracture or not. The model is
given images of x rays with labels, and it tries to continually update
its internal parameters to achieve the desired outcome.

One of the first parts of training is to prepare the data to be used.

This involves acquiring the data - for example, x ray images, and
preparing them to be used to train a classifier. Many times, data that
is gathered may not be suitable to be used, as it may be noisy, or, not
in the correct format. Another problem could be the fact that there is
much more data than needed. This could be a problem, because it would
increase the time taken to train the classifier, and can also
necessitate more computational resources. Therefore, some data may need
to be taken as a sample, from the overall dataset, to be used to train
the model. Care must also be taken to ensure that the images do not
violate any privacy laws, and must cover a variety of different
perspectives, to allow the model to infer the category of an unseen
image better.

The data may also need to be normalized, by, for example, scaling the
pixel values of the images. Scaling is important as it leads to better
convergence for the classifier. Convergence means that further
iterations of parameter updates of the classifier are unlikely to lead
to a vastly improved performance. Convergence is important as it can
help to signify when the process of training the classifier is complete
(mljourney, 2024)[^3]. For example, one way of scaling, in the case of
medical images, is to "transform" the pixel range from \[0, 255\] to
\[0,1\], which is typically done by dividing the pixel value by 255.
This is important, as it prevents the classifier from being dominated by
a single feature, which stands out more than others, leading to a more
balanced classifier (Kashyap, 2024)[^4].

The data may also be augmented, by for example, flipping the images,
rotating the images, increasing the contrast of the image. Data
augmentation is a crucial part of the training phase as it helps to
prevent - or at least minimize, overfitting, which is when the model is
unable to generalize, and can't accurately identify the patterns in the
data, rather it has learnt the training data "too well". Some ways in
which I can minimize this is to use a large dataset, and ensure the data
is augmented, by, for example, making copies of each image in which they
are rotated, zoomed or contrast-enhanced, to give the model more diverse
data to train off of.

When training, the model aims to modify its parameters so that it can
make the most accurate predictions possible. It does this, by trying to
minimize a loss function. This function, is essentially a metric to
measure how far away the model's prediction was, from the true value --
in terms of medical x ray image classification, this is a measure of
difference between the probability, provided by a model, of an image
belonging to a certain class and the actual class it belongs to
(Sciencedirect.com, 2023)[^5]

The different types of Classification: Binary, Multiclass, Imbalanced

Binary Classification

Binary classification is a type of classification, in which the model
aims to categorize new data -- for example, medical X rays - into two
distinct categories. It is a binary "choice" between two different
classes, for example, whether a chest x ray shows that a patient has
pneumonia, or is normal.

Multiclass Classification

Multiclass classification is a type of classification where the model
aims to classify unseen medical x ray images, into more than two
categories -- for example, they may try to classify a set of x rays into
different classes, depending on the part of the body which they show --
for example, arm, leg, chest, or abdomen.

There are many different methods to train multi-class classification
models.

One Vs Rest:

In this method of classification, each label essentially has its own
"classifier", which aims to produce a probability of a feature map
belonging to a certain class, compared to all the others (so, the
probability of it being in that class, against not being in that class),
and so there are, for any number of labels, the same number of
classifiers, in the model. For example, if you wanted to train a model
to classify images between "Pneumonia", "Tuberculosis (TB)", or "Lung
Cancer", in One Vs Rest, there is a classifier for "Pneumonia vs Not
Pneumonia" and "TB Vs Not TB" and so on.

One Vs One:

In a One Vs One method of multi-class classification, each class is
compared to another class, so, taking the Pneumonia, TB or Lung Cancer
example from earlier, this means there would be a classifier for
Pneumonia vs TB, Lung Cancer vs TB, and Lung Cancer Vs Pneumonia. This
means that, mathematically, there is a classifier for each unique "class
pair" in the classification problem - there are K(K-1) / 2 classifiers
for K labels.

In medical imaging, there is a need to balance training times, and cost,
meaning there must be a consideration for how many classifiers to use -
or, in other words, how many different diseases you want to be able to
differentiate with just one model. In this case, a OVR classifier may be
better, as it needs to train less classifiers, meaning computational
complexity, and therefore time to train, and cost, are lower. However, I
believe the most crucial part of any image classification of anything
medical, is about accuracy. Since OVO compares one class to just one
other class, it leads to less of an impact of imbalanced classes (where,
for example, you have many images of, for example, pneumonia, compared
to TB), which leads to the model being able to distinguish each class
better.

The different Classification Algorithms and their Advantages and
Disadvantages

Naive Bayes

Naive Bayes is a classification algorithm which uses a probability
distribution to figure out the likelihood of a particular image being a
member of a certain class. Naive Bayes is frequently used for
classification and clustering.

|                                                                                                                                                                                                                                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Advantages                                                                                                                                                                                                                                                         | Disadvantages                                                                                                                                                                                                                                                                                                                                                                                                    |
| It requires less training data. This can be good for medical imaging as it means that diagnosis could be done earlier, which, would for example improve patient experience. This also could reduce the computational complexity of the classification.             | It often assumes all class labels are independent, which isn't really accurate to real life datasets. For example, it may assume that the presence of a Pneumonia label, is unrelated to the presence of another label, but in real life, diseases are often related. This means it could see features that are indicative of multiple diseases and only consider one of the possibilities more than the others. |
| Fast compared to other algorithms                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Training and classification can be done with only one pass -- "epoch" through the data. This is much more efficient compared to other techniques such as CNNs, which often use multiple passes through the dataset to accurately learn filters to classify images. |                                                                                                                                                                                                                                                                                                                                                                                                                  |

Decision Trees

Decisions trees are an algorithm used in some classification problems in
which there are sets of decision nodes, and branches and leaves. Leaves
indicate classification labels, such as "Pneumonia" if you were looking
at a chest X ray, and branches indicate the feature combinations that
lead to such leaves. To improve performance, decision trees are pruned
-- which is essentially removing nodes/paths that do not contribute to
the prediction of the class.

|                                                                                                                                                                                                                               |                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Advantages                                                                                                                                                                                                                    | Disadvantages                                                                                                                         |
| Non parametric -- so it makes no assumptions about the number of parameters so in terms of medical imaging, it means it can pick out small variations in images to identify something clinicians may not be able to easily do | Prone to overfitting as they don't have a built-in mechanism to stop splitting, so they can't generalize to classify new unseen xrays |
|                                                                                                                                                                                                                               | Can be highly time consuming to train                                                                                                 |
|                                                                                                                                                                                                                               | If there is an unbalanced dataset, the decision tree can be biased towards the class with the most, in this case, images              |

Support Vector Machines

Support vector machines aim to classify data by finding the plane which
maximizes the difference between data points of the classes. First
features are extracted from the images and are then "plotted" onto a
feature space. The SVM algorithm then tries to find a hyperplane, which
is essentially a decision boundary between the two classes.

Kernel functions

As discussed previously, the overall goal of SVM, is to find a
hyperplane (straight line), to separate the different classes of data --
for example, arm X rays from leg X rays. However, when plotted onto the
feature space, many real world applications, such as x rays, may not be
linearly separable. Kernel functions are mathematical functions which
map this feature space to a "higher dimensional" space, which may make
it easier to separate the data and establish a hyperplane (Jain, 2024)[^6].

|                                                                                                                                                                  |                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Advantages                                                                                                                                                       | Disadvantages                                                                                                                       |
| Good for High Dimensional problems, such as medical xray images                                                                                                  | Can be harder to deal with larger datasets, due to it taking longer to train, or how hard it is to find a suitable kernel function. |
| Generalization is used in SVMs, which reduces the chance of overfitting, which means that the classifier should be more accurate when encountering unseen images | Can struggle with noisy datasets                                                                                                    |
| Work well when there is a larger separation in the data -- so classifying xrays from two different parts of the body                                             |                                                                                                                                     |

Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a type of neural network which
are designed to process data arranged in a grid -- for example, an array
of pixel values of an x ray image. CNNs are able to learn features and
patterns in images, such as edges, with convolutional layers. Each
neuron in this network, takes an input - for example, an array of pixels
of a section of an image, and performs a dot product, after which it is
followed by a non linearity function, such as "ReLu", which helps to
maintain the complexity of the CNN, which allows it to learn more
intricate features (Keldenrich, 2021)[^7]. CNNs also contain pooling
layers, which are responsible for reducing the size of the feature map,
while retaining the most important features (Wikipedia, 2025)[^8]. In
terms of medical x ray image classification, this means the network will
be able to classify images that are more similar to each other -- e.g.
chest x rays into either pneumonia or not pneumonia or another disease.

|                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Advantages                                                                                                                                                                                                    | Disadvantages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| No manual feature engineering needed, as the convolutional layers in CNNS are capable of learning filters to extract needed features automatically                                                            | For each, eg, 5x5 section in an image, a perceptron is needed, which, for an RGB image with 3 channels, has overall 5 x 5 x 3 weights -- 75. This means that the number of weights will increase substantially if you have a larger image -- for example, for a 100px x 100px rgb image, you will need around 100 x 100 x 3 -- 300,000 weights. This can significantly increase computational complexity, which increases the time to train, and it can lead to more overfitting as more filters / weights leads to higher complexity, which means the model can learn more intricate patterns, but has more capacity to perhaps learn more irrelevant features about the images. (Salehi et al, 2023)[^9] |
| High Levels of Accuracy -- The VGC16 CNN model achieved a 92.6% accuracy on the ImageNet dataset -- a dataset of RGB images containing 14million images, of 1000 different classes (GeeksforGeeks, 2020)[^10] |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

The metrics to Evaluate A Classification algorithm

Accuracy

Accuracy is the number of accurate predictions made by the model divided
by the total number of predictions possible -- essentially the size of
the dataset (Vujovic, 2021)[^11]. Suppose a model aims to classify x ray
images between classes of "arm", and "leg". The accuracy will be the
number of number of times the model correctly classified an image
labelled "arm" as "arm" and "leg" as "leg", divided by, this total, plus
the amount of times the model classified the x rays incorrectly. If, in
this example, Positive is an "arm" x ray, and negative is a "leg" x ray,
accuracy can be written as:

(TP + TN) / (TP + TN + FP + FN)

Accuracy, while useful as a general measure of the effectiveness of a
classification model, can be misleading if you have an imbalanced
dataset (for example, many more images of leg x rays than arm x rays),
as the model could predict the class with more images and naturally end
with a higher accuracy score.

Precision

Precision is the number of true positive predictions -- so, taking the
example from earlier, if "arm" is Positive, and "Leg" is negative, when
classifying x ray images, it is the number of times the model correctly
classified an "arm" image as an arm, divided by the true positives,+the
false positives -- in other words, the number of times the model
(incorrectly) classified a leg x ray as an arm x ray.

Precision is better than accuracy for imbalanced datasets, as it only
looks at one class -- the "positive" class, or, in the example's case,
the "arm" image class, rather than considering both classes. This leads
to it being less sensitive to imbalanced datasets, which may be more
common in the medical context, as some parts of the body get x rayed
more than others.

Precision = TP/TP + FP

Recall

Recall is the number of true positives divided by the total number of
positive predictions. If the "arm" class is positive and the "leg" class
is negative, it is the number of times the model correctly predicted an
arm image, as an arm, divided by this, plus the number of times the
model predicted a leg class, for an image that was actually an "arm"
image.

Recall = TP / TP + FN

F1 Score

F score, essentially combines both precision and recall, and is defined
mathematically as:

(2 x precision x recall) / (precision + recall)

F1 Score has many of the same advantages as Precision and Recall, when
compared to accuracy. A score closer to 1, is considered better.

Choice of Dataset

When choosing a dataset, for a medical image classifier, care must be
taken, to ensure the images are labelled accurately. If there are any
mistakes in the dataset, it can affect the accuracy of the model, which,
if used in a real hospital, can lead to certain conditions not being
identified, or being mis-identified, which is a critical problem. In the
section below I will compare some datasets and discuss which I will use
for my project.

NIH Chest Xray 8:
<https://www.kaggle.com/datasets/nih-chest-xrays/data/data>

|                                                                                                                                                                                                                                                                      |                                                                                                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Advantages                                                                                                                                                                                                                                                           | Disadvantages                                                                                                                               |
| Very large dataset- 112,120 images from 30,000 patients. Prior to this dataset, the largest x ray dataset was openi with just over 4000 images -- this will allow for the model which it is trained on, to be much more accurate and allow it to learn more patterns | NLP Extracted labels -- could be some erroneous labels, but checking this with 112,120 images is very time consuming and resource intensive |
| Despite the NLP Extracted labels, accuracy is stated to be \>90%                                                                                                                                                                                                     |                                                                                                                                             |

This dataset has 15 classes -- 14 disease classes, and one "no
findings", so will be a multi-class classification problem.

ChexPert <https://www.kaggle.com/datasets/ashery/chexpert>

ChexPert is a dataset collected by the Stanford ML Group, and it
contains over 200,000 chest x ray images from Stanford Hospital, taken
between October 2002 and July 2017. Chexpert also has 14 classes, for 14
different diseases.

|                                                                                                                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Advantages                                                                                                                                                                                                                                                                                                      | Disadvantages                                                                                                                                                                                                                                                 |
| Large dataset, from many years, which may lead to naturally higher diversity in the dataset, which will allow the model to generalize better, leading to it learning better filters (in the case of CNNs).                                                                                                      | Heavily imbalanced -- for example, there are \~46000 images labelled as lung lesions, whereas over 100,000 images were labelled as "Lung Opacity" -- this suggests techniques such as Undersampling, or augmentation may be needed to "even out" the classes. |
| Such a large dataset may somewhat reduce the amount of data augmentation necessary, which will speed up time to train, and perhaps may lower the computational complexity and resources needed to train the model, which will be good in hospitals as they may not always have access to the fastest computers. |                                                                                                                                                                                                                                                               |

Project Plan Draft

Title

Automatic Classification of X Ray images

Description of the project

Classifying medical images has been heavily researched and trialled, as
being able to do so effectively would mean diagnoses would be able to be
done much quicker, resulting in much better patient outcomes. One
approach to this, is to use a Convolutional Neural Network (CNN), which
learn features from images, and use these to accurately classify images
into the desired classes or categories.

However, one problem with this, particularly when it comes to medical
images, is that the classes may be heavily imbalanced, due to different
parts of the body being X rayed more than others, and certain diseases
being x rayed more, as, for example, they were discovered earlier. This
means that, in a given dataset of chest x rays for example, there may be
many more images labelled as "pneumonia", compared to other classes such
as "Lung Opacity", which would mean that the model may be "biased"
towards the majority class, inaccurately labelling images as
"pneumonia", which, in the medical context, could be catastrophic.

This project aims to design, develop and evaluate a robust convolutional
neural network to classify chest x ray images, despite this class
imbalance, to properly classify unseen chest x rays. The dataset I have
chosen for this, is the ChexPert Small dataset from Stanford ML. The
ChexPert small dataset contains 224,316 chest x ray images, split into
14 classes. The dataset is also quite imbalanced, with 46,000 images
labelled as "lung lesions", whereas over 100,000 were labelled as "lung
opacity", making this dataset a good candidate to investigate the effect
of class imbalance on model performance.

This project does not come without its challenges, for example, the
issue of how exactly I address the problem of class imbalance is crucial
to the performance of my model. I will need to use augmentation
techniques such as rotating, scaling, and changing contrast to create a
more diverse dataset, and will need to consider other techniques such as
undersampling to balance the datasets. Another challenge will be trying
to find the right balance of complexity, and computational performance
when training the model. A more complex model, with more hidden layers
of convolution and pooling layers, may lead to better accuracy and a
lower false classification rate, however, I will need to balance this
with the physical constraints of the computing power I have available. I
will also need to consider the methods with which I evaluate the model
-- for example, accuracy, while an okay general indicator of
performance, can be misleading, as the model could get a high accuracy
score by just predicting the majority class, regardless of what the
actual image shows.

Main technical problem of the project

The main problem that my project aims to solve, is to design, develop,
and evaluate a robust CNN model which can accurately classify different
chest x ray images, despite the significant class imbalance present in
the dataset. If class imbalance is not properly addressed when
developing an x ray image classifier, it could lead to the model being
biased towards the majority class. This would mean that there is a
higher chance of misdaignosis, which is an extremely big issue if it
happens in the medical field. The project will consider all aspects,
from data preparation, to the complexity and depth of the model itself,
and the use of a variety of evaluation metrics to create a robust,
accurate medical x ray classifier, to help with diagnosing certain
conditions quicker.

Indicative Timeline

October 16 - Nov 10^th^:

- Research imbalance further -- ways to deal with imbalance -- e.g.
  augmentation
- Research implementation methods, libraries etc, decide on methodology

Nov 10^th^ - Nov 20^th^:

- write up interim report

Nov 21 - \~Dec 10^th^

- Implement baseline model

\~Dec 11^th^ - February 20^th^:

- Iteratively improve the model, adding and experimenting with different
  augmentation techniques (e.g. rotation, scaling, adjusting contrast
  etc), changing number of layers, experimenting with different
  optimization functions
- Implement the evaluation algorithms (e.g. F1 Score, Precision, Recall)
  and evaluate the model

9^th^ february - 20^th^ Feb

- interview preparation
- attend interview

March -- April 25^th^:

- Write dissertation, and improve it over time

<!-- -->

- Prepare viva/presentation
- Final touches on software system

April 25 to April 27^th^:

- Check all components and get them ready for submission

April 27^th^:

- Submit all necessary components

Paper Summary:

Common Feature Extraction Techniques

Feature Extraction is a method to try to extract the most relevant
features of an image, and assign them to a class label, such that these
features can be recognized when a model "sees", a new image.

There are many different features that can be extracted from images.
These include colour features, texture features, and shape features.

Ways to Extract Colour Features

Colour Histograms

One of the most common methods to extract colour features from an image,
is known as the colour histogram. The colour histogram, represents the
distribution of colour in an image. The colour histogram achieves this
by separating the different colour channels into bins, which are
"ranges" of different colours. For example, one bin may be holding
pixels with an R value (if we are using the RGB colour space), between 0
and 84, another bin may hold the R values between 85 and 169, and
another may hold values from 86 to 255. This is then repeated for each
of the other colour channels -- in an RGB image, this would be Green and
Blue. Then, for each pixel in the image, the count of the bin
corresponding to the pixel values of a given pixel is increased. For
example, if a pixel in an image has RGB Values (60, 100, 250), then the
count of the Red bin holding values between 0 and 84, the Blue bin
holding values between 86 and 169, and the green bin holding values from
170, to 255 are incremented. The end result will produce a graph which
shows the count of pixels which had colour ranges from each bin.

Colour histograms may be advantageous if we are comparing things which
are, in terms of the colours, very different. However, when the colours
are very similar, colour histograms are not the best method, as it
ignores the shape of an image -- so, for example, a chest x ray of
someone with Pneumonia, versus someone without, may look very similar in
terms of the distributions of their colour, but actually show very
different things.

Colour Moments

Colour moments are used to try and "summarize" the colour distribution
in an image. There are 3 values (moments), which are used to show this
-- mean, standard deviation, and skewness. The mean colour moment aims
to show the average Red, Green and Blue values in an image. This is not
all that effective on chest X Ray images, as, by the very nature of the
images, they are likely to have similar means.

The second measure is the standard deviation colour moment. This is
defined as the square root of the variance of the distribution (Mutlag
et al., 2020)[^12], and this means that it shows how far the colour
values in an image, are from the mean -- if the standard deviation
moment is lower, it means that the colour values in an image are similar
to one another. Similarly to the mean colour moment, this is not too
helpful with X Ray images, as these are all typically gray scale, and so
have little meaningful variation in colour, meaning the information
gained is not too helpful when it comes to classifying these images into
different classes.

The third measure is skewness, which aims to show the shape of the
colour distribution, meaning it shows whether the distribution of the
colours across the image is skewed to the left (lower values -- darker)
or right (higher values, lighter), of the mean.

Texture Features

Texture features can give lots of information about the make-up of an
image, and these features contain information about the contrast, the
number of boundaries present -- areas in an image where the pixel
properties (e.g. colour) change abruptly etc. In image processing, there
are two concepts in texture features -- tone, and texture itself. Tone
is when, for example, a patch of an image has little variation of
features, the "dominant" property is tone, but if it has a wide variety
of features, the dominant feature is texture (Textural features for
image classification, 1973)[^13].

_Some Ways to Detect Texture Features_

Local Binary Patterns

Local Binary Patterns (LBP), is a method to extract texture features
from an image, in which a given pixel is compared against its
neighbours. If this neighbour has a higher intensity than the central
pixel, then a value of 1, is assigned, and if the neighbour has a lower
intensity than that of the central pixel -- the "threshold", then a
value of 0 is assigned. This process is repeated for each neighbour in
the region. Imagine a 3x3 grid across a part of the image, encompassing,
for example 9 pixels. The pixel at "index" \[1,1\] (if starting from 0),
is considered the central index. All pixels around it, \[0,0 0,1 0,2
1,0... 2,2\] are compared and assigned a value, depending on if they
have a higher intensity than the central pixel. Then, these binary
values (1, or 0) are combined to create a binary code for the central
pixel, which is then converted to a decimal value (Ihalapathirana, 2023)[^14].

![](Pictures/100000010000032E000001CF2B411B00139B416B.png){width="6.505cm"
height="3.701cm"}

<table>
<tbody>
<tr class="odd">
<td>Advantages of LBP</td>
<td>Disadvantages of LBP</td>
</tr>
<tr class="even">
<td><p>Can highlight abnormal areas of the xrays, by measuring pixel
differences, so subtle changes in the xrays can be seen by used by the
classifier or physicians to accurately classify and diagnose
diseases</p>
<p>Relatively Easy to compute</p></td>
<td>Easily affected by noise in the images – not ideal for x rays as
they can have noise eg grainy textures</td>
</tr>
<tr class="odd">
<td></td>
<td></td>
</tr>
</tbody>
</table>

GLCM

GLCM (Gray Level Cooccurrence Matrix) is a method of extracting spatial
relationships between pixels. This method of extracting texture features
aims to show how often pairs of pixels, with two intensity values occur
together, given a certain distance and "offset", which is a position
operator applied to a pixel, to reach another -- an example offset could
be "one right, one up", in simple terms, for example. The amount of
times a pixel with a certain intensity appears with another pixel with
the given offset is counted, and combined to form a N x N matrix, where
N is the number of "gray levels" -- for an 8 bit image, grayscale image
the gray value can take on any value from 0-255 meaning the resulting
GLCM Cooccurance Matrix would be a 256x256 Matrix. In this matrix, the
position corresponds to the pixel intensities, and the value corresponds
to the frequency it appeared. For example, in the matrix, position (0,0)
contains information about how often a pixel with 0 "gray level",
appears next to another pixel with 0 "gray level" at the given offset.

|                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Advantages                                                                                                                                                                                                                                          | Disadvantages                                                                                                                                                                                                                                                     |
| Because it essentially analyses the relationship of the intensities of pixels and their orientation / direction to one another, it can help to detect more subtle differences in e.g. an area of unhealthy tissue next to an area of healthy tissue | Computationally Expensive                                                                                                                                                                                                                                         |
|                                                                                                                                                                                                                                                     | Struggles near class borders -- this means it may not be suitable for extracting features from different chest x rays, as the images for e.g. a healthy xray, and one showing pneumonia are largely similar                                                       |
|                                                                                                                                                                                                                                                     | Ensuring you select a correct distance is crucial, as if it is too small, you may lose information about the texture pattern as a whole, but if it is too large, or it may just capture global texture properties, rather than in a specific section of the image |

Scale Invariant Feature Transform

Scale Invariant Feature Transform (SIFT), is a method of feature
extraction which aims to extract "key points" from an image, which it
then uses, to recognize other instances of these keypoints in new
images. Keypoints are unique areas of an image, that can be used to
recognize objects in other images, regardless of if the other image is
scaled, or rotated. The SIFT algorithm works by first creating images of
different sizes, and applying the Gaussian kernel to it, which results
in a blurred image. It applies this filter, causing blurring, at
different "intensities", to multiple versions of the image, This results
in "octaves", of the same image, where each image is the same size, but
each image has a different level of blurring. Then a process known as
"Difference of Gaussian" occurs. This means that, two images, of
different scales (in terms of the level of blurring), within the same
octave (the same size) are subtracted from one another, to create a
"Difference of Gaussian" (DOG) image. It is this which causes SIFT to be
invariant to scaling, such as increasing the size of an image. Keypoints
are then identified between these images, by comparing a pixel of an
image in one scale, with its neighbours, and that same pixel location,
in images of a higher scale (more blurred images of the same size), and
lower scales (less blurred images of the same size). To designate this
image as a potential keypoint, it must be a local extreme. This means
that this pixel's value is either greater than, or less than, all of its
neighbours across the scales. Once keypoints have been identified, they
are given an orientation (Ajmera, 2024)[^15]

|                                                                                                 |                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Advantages                                                                                      | Disadvantages                                                                                                                                                                                                                       |
| Less affected by noise -- good for medical images, which may have a significant amount of noise | Struggles in low light / low contrast conditions -- this could be bad for medical images, as x rays are usually low contrast -- so for example, finding a keypoint to distinguish an area with pneumonia in a lung can be difficult |
| Can generate many features from small objects                                                   | Computationally expensive, so may not be economical for future use with hospitals                                                                                                                                                   |
|                                                                                                 |                                                                                                                                                                                                                                     |

[^1]:
    > Alnuaimi, A.F.A.H. and Albaldawi, T.H.K. (2024). An overview of
    > machine learning classification techniques. Bio web of
    > conferences/BIO web of conferences, 97(4), pp.00133--00133.
    > doi:https://doi.org/10.1051/bioconf/20249700133.

[^2]:
    > Wang, Y. and Kosinski, M. (2018). Deep neural networks are more
    > accurate than humans at detecting sexual orientation from facial
    > images. Journal of Personality and Social Psychology, 114(2),
    > pp.246--257. doi:https://doi.org/10.1037/pspa0000098.

[^3]:
    mljourney (2024). *What is Convergence in Machine Learning? - ML
    Journey*. \[online\] ML Journey. Available at:
    https://mljourney.com/what-is-convergence-in-machine-learning/.

[^4]:
    Kashyap, P. (2024). *Image Normalization in PyTorch: From Tensor
    Conversion to Scaling*. \[online\] Medium. Available at:
    https://medium.com/@piyushkashyap045/image-normalization-in-pytorch-from-tensor-conversion-to-scaling-3951b6337bc8.

[^5]:
    Sciencedirect.com. (2023). *Training Phase - an overview \|
    ScienceDirect Topics*. \[online\] Available at:
    https://www.sciencedirect.com/topics/computer-science/training-phase.

    ‌

[^6]:
    Jain, A. (2024). *SVM kernels and its type - Abhishek Jain -
    Medium*. \[online\] Medium. Available at:
    <https://medium.com/@abhishekjainindore24/svm-kernels-and-its-type-dfc3d5f2dcd8>.

[^7]:
    > Keldenich, T. (2021). Easily understand non-linearity in a Neural
    > Network. \[online\] Inside Machine Learning. Available at:
    > https://inside-machinelearning.com/en/easily-understand-non-linearity-in-a-neural-network/ > \[Accessed 5 Oct. 2025\].

[^8]: Wikipedia Contributors (2025). *Pooling layer*. Wikipedia.

    ‌

[^9]:
    Salehi, A.W., Khan, S., Gupta, G., Alabduallah, B.I., Almjally,
    A., Alsolai, H., Siddiqui, T. and Mellit, A. (2023). A Study of CNN
    and Transfer Learning in Medical Imaging: Advantages, Challenges,
    Future Scope. *Sustainability*, \[online\] 15(7), p.5930.
    doi:https://doi.org/10.3390/su15075930.

    ‌

[^10]:
    GeeksforGeeks (2020). *VGG16 \| CNN model*. \[online\]
    GeeksforGeeks. Available at:
    https://www.geeksforgeeks.org/computer-vision/vgg-16-cnn-model/.

[^11]:
    Vujovic, Ž.Ð. (2021). Classification Model Evaluation
    Metrics. *International Journal of Advanced Computer Science and
    Applications*, 12(6).
    doi:https://doi.org/10.14569/ijacsa.2021.0120670.

    ‌

[^12]:
    Mutlag, W.K., Ali, S.K., Aydam, Z.M. and Taher, B.H. (2020a).
    Feature Extraction Methods: A Review. _Journal of Physics:
    Conference Series_, 1591, p.012028.
    doi:https://doi.org/10.1088/1742-6596/1591/1/012028.

[^13]:
    > _Textural features for image classification_ (1973).
    > https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4309314.

[^14]:
    Ihalapathirana, A. (2023). _Understanding the Local Binary
    Pattern (LBP): A Powerful Method for Texture Analysis in
    Computer..._. \[online\] Medium. Available at:
    https://aihalapathirana.medium.com/understanding-the-local-binary-pattern-lbp-a-powerful-method-for-texture-analysis-in-computer-4fb55b3ed8b8.

[^15]:
    Ajmera, G. (2024). _Feature Extraction of Images using GLCM (Gray
    Level Cooccurrence Matrix)_. \[online\] Medium. Available at:
    https://medium.com/@girishajmera/feature-extraction-of-images-using-glcm-gray-level-cooccurrence-matrix-e4bda8729498.
