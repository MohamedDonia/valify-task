# Sign Language Classifier

With sign language, we can provide a way of communication without the use of voice.  This is very 
importantfor individuals who have impaired hearing and inability speaking because it provides them 
an alternativemeans of communicating with other people.  Furthermore, sign language is an universal 
language that canallow people of different speaking language to communicate.To understand sign language
is challenge since it requires memorizing a lot of hand poses and gestures.  somesigns even have very 
similar appearance.  So we need an automatic sign language recognition system allowinganyone to 
understand sign languages.


In this repo, we will go through dataset and inspect it, train different models on the datatset to 
compareresults, perform cross validation and deploy model to put into production.


## Dataset

The  dataset  used  was  based  on  the  American  Finger  Spelling  format. 

<img src="https://github.com/MohamedDonia/valify-task/blob/main/assets/hand-pose-alphabet.jpg" alt="drawing" width="300"/>

The dataset used in training the network was composed of static hand poses of letters form the alphabet,each character consist of 3000 images of letters formed by the hands.  Samples of training images.

For Training, we will create data.csv file that contains path of the images and corresponding labels.  Also wewill create alphabet dictionary.json file that contain mapping from character to labels.

Dataset will get high accuracy for both training and validation data but in real life it may not work properly for the following reasons:
- Although  every  alphabet  in  dataset  consist  of  3000  images  but  these  images  looks  very  similar  asdataset taken for 1 person.
- Dataset taken with only 3 backgrounds and this not enough to generalize for the real life case.

<img src="https://github.com/MohamedDonia/valify-task/blob/main/assets/sample-of-dataset.png" alt="drawing" width="300"/>

## Model Architecture

Begin  training  with  as  a  baseline  then  we  can  up  scale  or  down  scale  model  to  get  a  better  accuracy  orminimum  model  parameters.   Residual  Network  (ResNet)  is  undoubtedly  a  milestone  in  deep  learning.ResNet is equipped with shortcut connections between layers,  and exhibits efficient training using simplefirst order algorithms.

We performed training with the following specs:
- Splitting  dataset  to  5  folds  with  stratification  Kfolds  to  ensure  uniform  distribution  of  labels  for  5folds4.
- Using the following augmentations:
  - Rotation with 10 degrees and P=[0.4, 0.6]
  - Horizontal flip with P=[0.5, 0.7]
  - Random brightness contrast with P=[0.4, 0.6]
  - Channel shuffle with P=[0.3, 0.5]
  - Hue saturation value
  - Blur with P=0.5 and BlurLimit = [3, 7]
  - Coarse dropout with P=0.5, MaxHoles = 10, MinHoles = 8, MaxDim = (30, 30) and MinDim =(10, 10)
  - Resize(244, 244)
- 64 batch size.
- 10 epochs.
- Using one cycle learning rate to increase learning speed.
- Applying early stopping criteria to stop training in case of validation loss not imporoved.
- using AdamW optimizer which improves Adam’s generalization performance, allowing it to competewith SGD with momentum on image classification datasets.

After running training script for the 5 folds with 10 epochs every fold using early stopping callback, we getthe following results for both training and validation:

Fold | #1 | #2 | #3 | #4 | #5 | CV
--- | --- | --- | --- |--- |--- |---
Validation | 0.99982 | 1 | 0.99982 | 0.9997 | 0.9998 | 0.9998

## Reduce Model Architecture
### MobileNetV2

After  training  model  with  ResNet18  and  get  CV  accuracy  0.9998  ,  we  can  reduce  network  size  to  getsmall  model  and  keep  a  high  accuracy.   The  MobileNetV2  architecture  is  based  on  an  inverted  residualstructure where the input and output of the residual block are thin bottleneck layers opposite to traditionalresidual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwiseconvolutions to filter features in the intermediate expansion layer.

After running model with the same configurations of ResNet18 but with only 1 fold , we find that the theaccuracy remains the same with lower learning rate but with a greet reduction in model size and less floatingpoint operations per second (FLOPS). We find the loss and accuracy over training and validation data areplotted in the following 2 figures, we found that the accuracy nearly 0.995228

### queezeNet 1.0

After training model with ResNet18 and get CV accuracy 0.9998 and training MobileNetV2 with 0.9952 val-idation accuracy, we will reduce model further to maintain same accuracy with very tiny model.  SqueezeNetachieves AlexNet-level accuracy on ImageNet with 50x fewer parameters [1].  with only 3 Mb we can achieve0.994 accuracy in validation, although we changed some parameters to get this results, we increased max-imum learning rate to 1E-4 in our scheduler and increased number of epochs to 20, maintaining all otherhyper parameters the same including augmentation parameters.

Comparing 3 models performs from the view of model size,  accuracy and FLOPS and inference time areshown in the following table

Model | CV | Modelsize (Mb) | Parameters | FLOPS | Speed(ms)
--- | --- | --- | --- |--- |---
Validation | 0.99983 | 44.8 | 11,191,389 | 1.826 GFLOPS | 4.693
Validation | 0.99522 | 9.3 | 2,261,021 | 332.98 MFLOPS | 11.232
Validation | 0.9941 | 3 | 750,301 | 724.9 MFLOPS | 4.53
