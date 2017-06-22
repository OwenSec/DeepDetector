To craft FGSM adversarial examples for ImageNet, [Caffe](https://github.com/BVLC/caffe) environment (including [pyCaffe](http://caffe.berkeleyvision.org/installation.html)) should be correctly configured. Once it's done, you may run detecting_ImageNet_examples_Crafted_By_FGSM.py without any hindrance. For those who are familiar with [Jupyter](http://jupyter.org/), a jupyter notebook file is also provided.

##### NOTE: To enable backpropagation for gradient information of an image, 'force_backward: true' must be inserted to deploy.prototxt before the first line, which is 'name: "GoogleNet"'.
