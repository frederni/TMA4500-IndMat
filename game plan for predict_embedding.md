# Issues in embedding code

* How can the model extract features from the actual rows? Now it looks like it just tries to predict label based on IDs alone?
* Creating the Dataset object with LabelEncoded customer and article IDs take a long time; is it very optimizable?
* MAP12 compares predicted to true values so we have to round the predictions (\in 0,1) to the closest category. 