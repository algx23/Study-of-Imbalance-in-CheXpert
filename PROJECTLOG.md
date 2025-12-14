# Project Log

Please regularly update this file to record your project progress. You should be updating the project log _at least_ once a fortnight.

## Week 1 [w/c 29.10.2025]

- detail your actions here
- you can include details of commits if necessary
- alternatively, details of any reading you have completed in support of your project

- This week I set up an initial meeting with my supervisor, in which we discussed the expectations of the projet - we discussed that the project includes significant research, where I would have to read lots of papers and articles to build an understanding of classification both generally and in terms of medical images. My Supervisor and I also identified that my knowledge of the domain, at the moment, was relatively shallow, and I got some advice on some reading to do, to better understand Classification
- I spent the week reading up on classification and plan to do so for a week.
- I have also started writing up a literature summary "essay" in which I discuss the content of what I have read.
- Some of my reading involves:

  - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9335581&tag=1 -> this talks about the effectiveness of using Convolutional Neural Networks to detect Covid 19 from chest x ray images

  -https://www.bio-conferences.org/articles/bioconf/pdf/2024/16/bioconf_iscku2024_00133.pdf -> This gives an introduction to classification and the different algorithms and techniques used in classification - for example, CNNs, K nearest neighbor, Ensemble algorithms

  - https://www.mdpi.com/2071-1050/15/7/5930 which talks about some of the positives, and challenges of CNNs in ML contexts - such as how it does not require manual feature extraction, and how it is good for large datasets.

  - https://doi.org/10.14569/ijacsa.2021.0120670 which touches on the different evaluation metrics used in evaluating the performance of a model. Initially, in table form, i have touched on the advantages and disadvantages some of the main evaluation metrics used in image classification.

  - I will continue to read up on classification, continue to write up about what I have read, as this helps me to consolidate my learning, and prepares me well for the literature review/summary portion of the interim report. It also allows me to have a deeper understanding of the problem domain so I can think properly about which problem I will attempt to solve and the ways in which i go about it. I will also think about which dataset i want to use, as well as justifying why it is good, and begin to think more about my aims for the project, and my methodology in more depth.

## Week 2 [w/c 06.10.2025]

- I started to think about the problem statement that I wanted to write about. I decided on centering my project around designing, developing and evaluating a CNN Model despite class imbalance present in the dataset
- I also seeked feedback from my supervisor about the research I had done in week 1
- Last week I also decided on using the ChexPert dataset, and thought about the justification for this

## Week 3 [w/c 13.10.2025]

In this week I wrote the project log and thought more about my project objectives, and problem statement

## Week 4 [w/c 20.10.25]

- Last week I started to read papers about methods of augmenting data - https://doi.org/10.1186/s40537-019-0197-0
  - This paper talks about the different types of data augmentation, such as flipping, cropping, rotations etc
  - I also started reading this paper: https://doi.org/10.1016/j.neunet.2018.07.011 which talks about comparing methods of dealing with class imbalance
- This week, (wc 28.10.2025) I will write up the summary of the second paper listed, and also review different feature extraction techniques, as suggested by my Supervisor, and prepare for a group meeting on 30.10.2025


## Week 5 [w/c 27.10.2025]

- I read a study about different sampling methods and how it affects model performance
- I also had a meeting with my supervisor, for which I prepared a presentation, and presented my progress so far
- As per feedback I will read some papers about intra-class variance and inter-class similarity, as it is also a problem I will have to deal with, as all images are chest X-rays and so are largely similar
- Going forward I will also start planning how to write my interim report

## Week 6 [w/c 3.11.2025]
- In this week I started structuring my reading into different categories (augmentation/dealing with imbalance, papers detailing different CNNs and their performance on a variety of datasets, evaluation methods), so that they could be used for the interim report literature survey
- I also thought about and made notes about the different Aims and Objectives of my project, and started making notes for the structure of my interim report
- I also thought about the potential architecture for my CNN - 4-5 layers, with ReLu Activation functions and pooling layers
- Found a paper comparing a model of CNNs with dropout, and without dropout [https://ieeexplore.ieee.org/abstract/document/9057809] so I also considered potentially using dropout in my CNN
- During w/c 10.11.2025 (this log entry is written on 11.11.2025 retrospectively for w/c 3.11.2025) I will start writing the initial draft of the Interim Report, to hopefully finish by the end of the week and send to my supervisor for feedback

## Week 7 [w/C 10.11.2025]
- During week 7, I started writing up my initial draft of the interim report
- i finished it on thursday 13.11.2025, at which point i sent it to my supervisor for feedback
- following the feeedback - that i also need to consider inter-class similarity, and intra-class variance -  I started improving it to address the feedback, which i am continuing to do in [w/c 17.11.2025]

## Week 8 [w/c 17.11.2025]
- This week I Improved the interim report further, based off the feedback, and addeed a section about my methodology, and some expected outcomes, as my project does involve some experimentation and comparing resuts

## Week 10 [w/c 1.12.2025]
- This week I spent time downloading the dataset and studying the structure of the Train CSV
- In the train CSV there are rows for each image, and each image has both a Frontal, and Lateral orientation
- Each Disease can have 3 values: 1 (Positive - the image depicts the condition), 0 (negative - the image does not depict this class label), or -1 (the presence of this class label is uncertain)

## Week 11 [w/c 8.12.2025]
- This week I set up the initial project file for preprocessing images, and tested the environment works
- I also set up google colab, with my VSCode, so that I can train models and develop, without a powerful dedicated GPU
- Progress has been slower than expected, due to having other assignments due, however, now during the christmas break, I aim to get fully back on schedule
- The plan for the next week, [w/c 15.12.2025] is to load the initial dataset, sample a subset and start to build the initial model.
