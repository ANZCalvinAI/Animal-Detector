# Schedule

Help to have a clear view on this R&D project.

[According to Taxonomy Australia, there are more than **450,000 species of animals** in Australia](https://en.wikipedia.org/wiki/Fauna_of_Australia)[1](https://en.wikipedia.org/wiki/Fauna_of_Australia)

**850** species of birds, and according to Taxonomy Australia, the current best estimate is that there are more than **200,000 species of insects** in Australia. [Only around **62,000** of these have been named so far. [The Australian Insects Website states that there are currently around **86,000 species** identified in 661 families](https://www.australian-insects.com/).



### To-do Details

These are details for to-do list.

#### Detection

- [x] Yolov5 detection model by Pytorch is ready.
  - [ ] Yolov8 or other recent better detection models as candidates. **

- [x] OpenCV DNN code loads the Yolov5 model for inference/prediction.



#### Classification

- [x] Create scripts for iNat2021 dataset analysis - docs/species.md
- [ ] Validate classification models.
  - [x] Validate ResNet101 for 500 classes.
  - [ ] ~~Validate ResNet101 for 500 classes based on roi images.~~
  - [x] Validate ConvNeXt for 500 classes.
  - [x] Validate EfficientNetv2 for 500 classes.
  - [ ] Validate EfficientNetv2 for 500 classes based on roi images.
  
- [ ] Validate detection models for 500 classes.
  - [ ] Validate Yolov5 for 500 classes.

- [ ] Split date of Insect (Insecta) and Bird (Ave) for 2-stage inference.
- [ ] Load Yolov5 to generate Roi smaller training images for classification.
- [ ] Split into 5 training datasets for 2700 species of insect.
- [ ] Identify the proper classification model, ResNet is just one of candidates.
  - [ ] Identify the deployment solution.
- [ ] Training model for 2nd stage inference.W
  - [ ] ~~Transfer learning.~~
  - [ ] Fine tune. **
- [ ] Training model for 1st stage inference.
  - [ ] ~~Transfer learning.~~
  - [ ] Fine tune. **
- [ ] Inference pipeline deployment on Web Scanner for testing.



## Contact

Jeffrey
