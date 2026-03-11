# Project Log

Please regularly update this file to record your project progress. You should be updating the project log _at least_ once a fortnight.

## Week 1 [w/c 29.9.2025]

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
- Progress has been slower than expected, due to having other assignments due, however, now during the christmas break, I aim to get fully backon schedule
- The plan for the next week, [w/c 15.12.2025] is to load the initial dataset, sample a subset and start to build the initial model.

## Week 12 [w/c 15.12.2025]
- This week I worked on preprocessing the dataset, to get it ready for training.
- I sampled the cheXpert dataset, from 200k+ images, to a stratified subset of 10,000 images, so that it would be easier to train the models, while still being representative of the class distributions and imbalance in the original dataset
- I spent time handling uncertainty labels, by initially deciding to make all labels with a value of -1 be 0
- I also started working on normalizing images for the dataset, using the PyTorch Normalize function, for which I wrote a function to calculate the mean, and standard deviation of my subset, to bring all pixel values of images from the range [0, 255] to ~[-1.8, 1.8] for faster convergence, and to help prevent overfitting

## Week 13 [w/c 22.12.2025]
- I also continued to work on normalizing images, by applying the normalizations to
  each image tensor in the dataloader
- This week I worked on creating the intial structure for baseline model class for
  training, and also did some minor refactoring work on the main class

## Week 14 [w/c 29.12.2025]
- This week was a little slower due to me having exams the next week [w/c 5.1.2025] for which i spent most of my time
- However I still made some progress on the project
- During week I fixed a critical bug in my training data where some labels were fully blank due to row indexing issues in Pandas
- I also focused more on testing the intial training of the model, and ensuring it would be ready for me to start the heavy experiment testing in the next few weeks
- I tested model training over more than 1 epoch and added a simple evaluate_model function to generate a classification report
- I also split the training set further into a validation set, of 1k images, meaning my train/validation split is 9000:1000
- I created a function to visualize the training loss over time in a graph using matplotlib
- I also started working on generating confusion matrices for each of the 13 classes

## Week 15 [w/c 5.1.2026]
- This week I started a little slowly, as I had an exam on 7.1.2026, however I made more progress afterwards
- In this week I worked on logging the metrics for each experiment, so that when i run it, i have the results stored for comparison
- I wrote the training loss which i worked on plotting last week, and validation losses, to files
- I also implemented a ValidationLossChecker class, to compute the validation loss after each epoch

- I also refactored my data-sampling script a little, to put the subset, train split, and vaidation split code into functions
- I also integrated this into main.py, checking if the train/validation splits exist, and creating them if not
- When I went to prepare the test dataset, I found that I could not access the official CheXpert test dataset easily, and it was also huge in terms of storage space
- To tackle this, I decided to use the valid.csv, a 234 image set, as my test set. The original CheXpert paper has validation results for 5 classes, though they compared label policies, so I will still be able to carefully compare results in my dissertation.

- I also worked on a baseline which would handle class imbalance, and plan to use this for all experiments - I created a function to calculate the class weights for each of the classes, and use this as the pos_weight parameter in the BCEWithLogitsLoss
- This meant that the model would penalize mistakes on rare classes more heavily, and since weighted loss seems to be a standard part of any CNN, I figured it would be good for the baseline to include it, though I plan to run an experiment of the baseline with and without weighted loss as well
- Towards the end of this week I ensured that the ValidationLossChecker would save the model with the lowest validation loss, and that this model would be used for evaluating the test set during the experiments.

## Week 16 [w/c 12.1.2026]
- This week I made a little less progress than I thought, as I was unexpectedly busy during the week
- However, I still made forward progress by for example, making the plot_loss function (renamed from plot_training_loss) to also plot the validation losses across each epoch, which would allow me to visualize how the model generalizes as it trains and also see at which epoch the model was best (i.e. validation loss doesn't improve for 10 epochs)
- I also did some minor refactoring work, by reworking the loss file saving logic into a function, and adding a save_model function to save the model
- In this week I also rectified a minor oversight where I was recalculating the mean and standard deviation for the test set. This was incorrect, as in reality, it is unlikely we have a big enough sample, or even all the data necessary to reliably calculate the mean/standard deviation. To fix this, I set the mean and standard deviation of the test data loader to be the same as the train data loader
- Additionally, I changed the validation dataloader to not have any transforms/augmentations meaning it is a more realistic unseen set

## Week 17 [w/c 19.1.2026]
- This week I started by updating the ValidationLossChecker to plot the Precision-Recall curve for each of the 13 classes
- When initially planning the project, I had just planned to have a separate branch for each experiment, and thought that I could just manually change the augmentations for each experiment
- However, after some thought, I realised that this might hurt the validity of the experiments, as I could make a typo or mistake when using the augmentations.
- Another consideration was time. If for example, I ran an experiment, and was away, and the experiment finished before I could get back, I would lose the "inbetween" time where no experiments ran.
- To address this, I spent the week working on a argument parser, so that I can pass in flags to specify augmentations. These can then be put into a powershell script, and then left to run.
- For example: py main.py --name baseline_model; py main.py --name baseline_model_with_class_weights --class_weights; py main.py --name baseline_with_class_weights_horizontal_flip --class_weights --hflip 0.5; etc

- I also found some logic bugs, such as the calculate_class_weights function not returning the class weights, which i spent time fixing.
- I also realized that the main files were getting rather large, and started planning a large refactor in the coming weeks
- I believe that I am in a very good position to start running experiments towards the mid-end of [w/c 26.1.2026]. While this is a lot later than the initial, interim timeline, I believe the use of the flags and shell script to define the experiments will help me gain back a lot of time.

## Week 18 [w/c 26.1.2026]
- In this week I started to think more about making my project better for the Principal Marker interview, on 10.2.2026 so I focused on adding help messages for the flags, as if the Principal Marker wanted to test my training script I thought it would help readability
- I also implemented a CLAHE flag `--clahe` to specify whether for a given run, the CLAHE augmentation would be used
- For this, I realized that it would be easier, in terms of readability and structuring my code to load the images as an OpenCV image, so I converted the ChexpertDataset class to do this.
- I also fixed some bugs, for example the PR curves would not save if EarlyStopping was not triggered, as I forgot to call the function to plot and save the curves. Additionally,I decided to save the evaluation logits themselves to files in case I wanted to calculate new metrics in future
- I also implemented a `--dropout` flag, similar to the `--clahe` flag, to specify whether dropout would be used in the model or not.
- I also had a meeting with my supervisor this week, to catch up on the status of the project, and he said that it would be good for the demo, to have something to show the models working - so I wrote a simple script allowing the user to provide a path to an image, and the name of an existing model, and the model would then run inference on that image and display the result.
- Towards the end of the week I also implemented a `--bn` flag, to specify whether a given run would have batch norm layers or not
- I also decided that, for the demo, I should refactor the code, so I started working on it in this week by creating all needed filepaths, before the main program flow starts

## Week 19 [w/c 2.2.2026]
- This week I continued on the refactor, and I started by encapsulating the training loop, and the evaluation loop into their own classes - Trainer and EvaluationLoop respectively. I also encapsulated the data loader preparation functions `prepare_data` and `prepare_test_data` into their own module
- Part of this refactor was also moving the report and confusion matrix creation logic, into their own MetricCalculator class 
- I continued by refactoring the PR curve computation logic into one loop, rather than having it loop over LABELS once for the graph containing all labels, and then again for the competition labels, I used two figures to loop through it once, and if the label was part of the CheXpert Competition, I would add it to both the complete and competition-only figures
- I also decided to compute the PR curves on the test set, as then it would be a more valid comparison between models - the validation set has been used to decide when to stop training, and so I thought comparing models with metrics calculated from this set, would not be scientifically valid.
- I also refactored the model settings - such as the augmentations, and the model name - to be stored in the constants file and then passed into main. I did this because it helped to make main a little cleaner, and meant that all the relevant "settings" for an experiment come from the same place which I think makes the code a little cleaner
- I also improved the formatting of all the code, and added a check to see whether an image in the ChexpertDataset class was none in the __get_item__ function, as if an image got deleted, the program would not exit gracefully
- I also added some logic to create comparison graphs based on the macro F1 Score, and increased the size of the test set to 2000.

## Week 20 [w/c 9.2.2026]
- This week I spent time making the final touches to get ready for my demo
- I added documentation to all of my code, as well as some small error handling
- However, after the demo, I had another assignment due, so most of my time was spent on that, but I have planned all experiments, and they will be run in [w/c 16.2.2026]. I ran 4 experiments for the demo but will run all of them in the next week

## Week 21 [w/c 16.2.2026]
- This week I spent some time running the individual experiments for each of the mitigation strategies
- I managed to do initial runs, for:
    - Baseline
    - Class Weights
    - Dropout
    - Rotation [5,10,45 degrees]
    - CLAHE
    - Color Jittering
- Next week I will work on running experiments with batch norm, and different combinations, after which I will spend some time analysing the results further - e.g. collecting per class results and putting the results in graphs
- Also, considering the feedback from my principal marker interview, I may add more augmentations, such as CutMix, and other loss functions, such as focal loss, though I need to read up more on this

## Week 22 [w/c 23.2.2026]
- This week, I continued the experiments as I added intial experimental results for horizontal, and vertical flips with a probability of the flip happening = 0.5
- Towards the end of the week, I thought that I could also compare different loss functions, as just comparing non-weighted BCE and weighted BCE felt like it was not enough
- I spent the week first reading the focal loss paper, and understanding what each part of the focal loss equation meant:
	- FL(x) = -alpha_t (1-p_t)^gamma * log(p_t)
		- alpha_t is the weighting factor: if the truth label is 1, alpha_t = alpha, otherwise alpha_t = 1-alpha
			and as mentioned in the paper, alpha can be set by inverse class frequency
		- pt is the probability of a given class: pt=p when y=1 otherwise pt=1-p
- Since the equation was given as a piecewise function in the paper, I spent some days in the week trying to derive / deconstruct the equation, in such a way that I could program it easily
	- I did this using Indicator functions, as they act a little like an if-else statement, so that I end with an equation where alpha_t and p_T are experessed only in terms of alpha and p
- This resulted in the following equation:
	- FL(x)=CE(αy(1−p)^γ+(1−α)(1−y) p^γ )
- When i write my dissertation, I think it would be good to have a small section going over this derivation, so that is the plan

- Focal Loss Paper: https://arxiv.org/pdf/1708.02002
- indicator functions: https://en.wikipedia.org/wiki/Indicator_function

## Week 23 [w/c 2.3.2026]
- This week rather than just thinking about the theory of focal loss I started to think how I could integrate this into the experiments
- One thing I realized is that, currently I was tracking validation LOSS as the primary metric for my Early Stopping, however, when I tested focal loss, the loss values were much smaller in magnitude than those given by the standard BCE Loss
- Because of this, I decided to switch to tracking the validation PR-AUC (which is essentially the average_precision_score in sklearn), as the scales would be similar across all experiments
- Additionally, I had some erorrs when I ran the initial Focal Loss Implementation, so I fixed these by making the class inherit from nn.module() which allows pytorch's backward pass for updating weights and gradients to work
- I also implemented the alpha-balanced focal loss, by adding a function to calculate the weights, by calculating the inverse class frequency of each class
- I also did some light changes to the comparison-graph-generating logic, by making the bars horizontal so they can all fit on the screen
- Additionally, I found, interestingly, in the test runs, that Focal Loss was not doing as well as i thought, and decided I could turn this into an opportunity to compare another loss function, and so started to research class balanced focal loss
- I spent some time reading the paper, linked below, and implemented CBFL, which uses a beta value to set the weights, rather than the raw inverse-frequency.
- After some reading, I found that this, in theory should be better, as it uses the Effective number of samples [(1-beta / 1-beta^ny)], which accounts for the fact that after you have so many samples, adding new ones doesn't really give much new information
- I then spent some time implementing the Class Balanced Focal Loss logic

- Another thought I had, while reading and implementing these loss functions, is that the probability of a given class would be lower,
perhaps due to focall loss forcing the model to learn more hard examples, and so its "confidence" would fall overall. This made me feel as though the 0.5 threshold i was using, was rather arbitrary, and, while it is good for comparing the raw ability of the techniques, in reality, no hospital going to use a model, without first tuning the thresholds for predicting a given class, to their data.
- To address this, I implemented logic to calculate the optimal thresholds for each class, selecting thresholds based on what would maximize the f1 score.

	
