Things I learned after implementating the UNet

* Even I do the upsampling, the size of height and width were not same in contracting path. So I have to crop the tensor to match the size.
* Author didn't use the same padding.
* Formulation of convolution output size
  outsize = (input - 1) * stride - 2* padding + kernal + output padding
* Two kind of segmentation, Binary segmentation which also called as Semantic Segmentation and Multi-class segmentation which also called as Instance Segmentation
* Author used cross-entropy loss function for training
* Segmentation task uses IoU(Intersection of Union) for evaluating the model. It means the model's performance is good if the value of IoU is close to 1.
* IoU used like accuracy in classification task. It calculated while each epoch and each epoch's validation IoU is calculated by averaging the value of several IoU of images. 
