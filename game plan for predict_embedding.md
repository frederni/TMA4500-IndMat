# Issues in embedding code

* How can the model extract features from the actual rows? Now it looks like it just tries to predict label based on IDs alone?
* Creating the Dataset osbject with LabelEncoded customer and article IDs take a long time; is it very optimizable?
* MAP12 compares predicted to true values so we have to round the predictions (\in 0,1) to the closest category. 
* Still not implemented: sample user never seen before if we train with 100% of dataset (how?)
  * Use a `prob` parameter that the negative sample should be from a 'new' user.
  * If `runif(0,1) < prob`, make random name and encode similarly to customer ID (or not, whatever)


TODO
* Try to pass a data sample without `LabelEncoder()` to the fastai implementation -- it should in theory work. I hope
* 