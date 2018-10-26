# Semantic Segmentation

The goal of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN)

[//]: # (Image References)

[image1]: ./misc_images/fcn_encoder_decoder.png
[image2]: ./misc_images/fcn.png
[image3]: ./misc_images/um_000004.png
[image4]: ./misc_images/um_000057.png
[image5]: ./misc_images/um_000083.png
[image6]: ./misc_images/umm_000033.png
[image7]: ./misc_images/uu_000098.png

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.


### Full Convoltional Network

Traditional CNNs are great for classification task for answering questions like "Is this a hotdog?", but its fully connected layers don't preserve spatial information, therefore cannot answer the question "Where in ths picture is the hot dog?"

CNN's fully connected layers have few disadvantages:

* don't preserve spatial information
* constrain the network's input size

FCNs can do what CNNs cannot do - it preserve the spatial information throughout the entire network, and will work with images of any size.

FCN is comprised of two parts: encoder and decoder.

![alt text][image1]

#### Encoder

Encoder extract features that will later be used by the decoder, which is similar to transfer learning. In this project, a pre-trained VGG model on ImageNet was downloaded and the input, keep probability, layer 3, layer 4, and layer7 are extracted. I did this in lines 21 to 44 in `main.py`.

#### Decoder

##### Transpose Convolutions

Decoder upsample images out of encoder such that the network's output is the same size as the original image. A series of transposed convolutions were used as following:

```
output = tf.layers.conv2d_transpose(input, num_classes, 4, strides=(2, 2))
```

##### Skip Connection

One effect of convolution or encoding is it narrows down the scope by looking closely at some pictures and lose the bigger pictre as a result. So even if we decode the outpt of the encoder back to the original image size, some information has been lost.

Skip connection is a way to retain information by connectin the output of one layer to a non-adjacent layer.

![alt text][image2]

The VGG-16 model already contains the 1x1 convolutions that replaced the flly connected layers. Additional 1x1 convolution layers were added to reduce the number of filters form 4096 to the number of classes, i.e., 2.

The scaling layers were added to the outputs of pooling layer 3 and 4.

I've added l2-regularization to all layers to avoid overfitting. 

I did this in the function `layers()` in lines 48 to 76 in `main.py`.

#### Classification and Loss

In this step I defined the loss to train FCN like CNN. I manuually added the regularization loss terms to the loss function.

I did this in the function `optimize()` in lines 80 to 96 in `main.py`.

#### Train Model

I trained the model in the function `train_nn()` in lines 100 to 131. The hyperparameters were chosen to be Epochs=50 and batch_size=5.

### Results

Here are some of the images from the otpt of the FCN with 2 classes: road or not road.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

### Future Workds

* Apply the trained model to a video
* Train and Inference on the cityscapes dataset instead of the Kitti dataset. 
* Augment images for better results