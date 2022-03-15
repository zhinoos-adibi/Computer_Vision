**This Repository is updating**

In this repository the concepts and codes of deep learning based in image processing will be reviewed.
some of the concepts that help to understanding of the subject will be explained.

Main Refrence:

COURSE: A deep understanding of deep learning

SECTION: Convolution and transformations

LECTURE: Convolution in code

TEACHER: Mike X Cohen, sincxpress.com

COURSE URL: udemy.com/course/dudl/?couponCode=202108



**Kernel concept:**

Kernel are filters that extract features from an image.The same kernel applied to different images will give different featuremaps. Use odd kernel sizes (3,5,7,etc)to have an exact center.

kernels are filters and thr pupose of filter ,which is the purpose of convolution colonel , is to extract specific features from an image.

Convolution is using a 'kernel' to extract certain 'features' from an input image. ... A kernel is a matrix, which is slid across the image and multiplied with the input such that the output is enhanced in a certain desirable manner.

kernels are generaly small(3 x 3 , 5 x 5 , 7 x7).In deep learning kernels learn the important features of the images.


فیلترهای مختلفی برای اعمال روی یک تصویر وجود دارند.برای مثال فیلتر برای حذف کردن نویزها یا فیلتری برای تشخیص لبه های تصویر دو فیلتر پرکاربرد می باشند.
در حالت کلی فیلترها به دو دسته خطی و غیر خطی دسته بندی می شوند .تمام فیلترها دارای یک کرنل هستند که کرنل ها در واقع یک آرایه ام در ان با مقادیر مشخص می باشند.مقادیر این آرایه کرنل و ابعاد آن باید به گونه ای طراحی شوند که در تصویر خروجی تصویر مطلوب ما ایجاد شود.به مقادیر عناصر کرنل ، وزن های فیلتر نیز گفته می شود.وظیفه ای که کرنل ها دارند این است که باید روی تصویر اصلی بلغرند و با محاسبات ریاضی ، تصویر خروجی را ایجاد کنند.

منبع :آموزش جامع پردازش تصویر با OpenCV ,Python

مولف : محمد حسن علیایی طرقبه




**what is pooling in deep learning?**

Pooling is nothing other than down sampling of an image. The most common pooling layer filter is of size 2x2, which discards three forth of the activations.
pooling is not formally part of Convolution , but pooling is often done immediately after convolution in deep learning.
Max pooling and mean pooling are common. Pooling also allows us to select feature and identify features in an image over broader spatial area and this is because pooling increases the receptive field size of the model .We actually apply pooling multiple time in a CNN model. And that's because as we go deeper into the CNN ,we want to have fewer pixels. Max pooling is useful because it highlights the sharp features. This is often use for increasing the constant when you have large difference in pixel intensity values between neighboring pixels.

In brief Max or Mean pooling?

Max Pooling :

Highlight sharp features.

useful for sparse data and increasing Contrast.

Mean Pooling:

Smooths images
Useful for noisy data and to reduce the impact of outliers on learning.

Now, these are kind of the clear cut cases where it's pretty straightforward which method you should prefer.


**Parameters of pooling**


**Spatial extent ("Kernel size"):**

The number of pixels in the pooling window. Typically set to 2(actually 2 x 2).

**Stride:**

The number of pixels to skip each window. Typically set to 2(produces no overlap.


Now this is the same concept as stride with convolution 

The benefit of having the stride and the spatial extent be the same number is that produces
 no overlap and no missing pixels.So we're not losing anything and we're not having any overlap.
 
 It is also possible to have the stride smaller than the kernel , and that creates some ovelapping wondows that is less common.


<img width="595" alt="image" src="https://user-images.githubusercontent.com/95547363/154861775-a6cfe899-fe3a-461d-9d31-60c0d0ddffc0.png">

![image](https://user-images.githubusercontent.com/95547363/154861909-b4c02e36-7243-4e78-8f4c-c8275ebc1d61.png)

In the below picture which stride is smaller than kernel we can see overlapping,that means it's possible to have some redundancy in these two values:

![image](https://user-images.githubusercontent.com/95547363/154862048-38e516ba-b1c9-4ef2-ba82-880718be1612.png)

**Transforming the image**


**The reasons to transform images**

Some times CNN's pre-trained model can apply image with special propeties .
Some times transform the image without changing the information of the image ficilitate the model training process.


**Creating and using custom DataSets**
You can imagine DataLoader like a box wich contains data.where these datas comefrom.
 
 to get the data into the data loader box ,we first transform the data into a data set.
 
 so  usually the procedure that we've been following is we start with our data in numpy format and then we have to transform it into a pytorch tensor.And then we transform that tensor into a dataset object , which contains a tuple of data with the data tensor and all of the corresponding labels also stored as tensor.And then we take this data set object and import that into the data loader.what do we do with the data in data loader?The most common thing that we do is we pull out minibatches.
 
 taking minibatch from the data loader box is a kind of like reaching your hand here and pulling out the small sample of data.Data loader contains all of the data and maybe we pull out only 16 data samples and corresponding labels.
 
 ![image](https://user-images.githubusercontent.com/95547363/155129776-9bb186d9-e9f9-46eb-b831-afec41697921.png)



![image](https://user-images.githubusercontent.com/95547363/155130020-2314919b-98b9-4b24-a16e-db0d78ef5506.png)

step one in two can be combined if you are importing the torch vision data set.

# Next Part

## COURSE: A deep understanding of deep learning


## SECTION: Understand and design CNNs


## TEACHER: Mike X Cohen, sincxpress.com

## COURSE URL: udemy.com/course/dudl/?couponCode=202108

------------------------------------------------------------------

**-Understanding and design CNNs(Convolutional neural networks)**

CNNs are very important in the development of image processing and computer vision

**Canonical CNN architecture**

Three types of layers in a CNN:

Convolution: learn filters (kernels)to create features maps through back propagition.


pooling : Reduce dimensionality and increase receptive field size.


Fully connected :prediction(categorical and/or continuous)




![image](https://user-images.githubusercontent.com/95547363/156172004-364e3ded-1534-4e93-b985-769513a705b8.png)



# CNN to Classify MNIST digits.

**Note :  MNIST Description**

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.




*What the architecture of CNN model looks like.**

![image](https://user-images.githubusercontent.com/95547363/156320619-c7a5feda-254e-44d7-826d-9b79d9949f29.png)

Input images are 28 X 28 with one channels as these are  grayscale images.

As you see in the above picture we have convolution -pool blocks and in the end we have fully concted layer.

Conv1 as you see in the picrure is (10 X 26 X 26) which 10 here corresponds to the number of features maps,which also correspond to the number of convolutional filter kernels that we will create.These are initialy set to random numbers and then they are learned and they are shaped through back propagition.

So 10 layers and 26 X 26 pixel means we are shaving off one pixel from the edge,so one pixel from the left , one from the right and so on top and bottom and that gives us 26 x 26.

Then we apply max pooling to this to the output of this convolutional layer thta preserves the number of feature maps.

we rduce the dimension in the individua unit in maxpool1 layer.

Now we go to conv2 layer(20 X 11 X 11) so twenty means we have twenty features maps or twenty convolutional kernels that the model is going to learn and then 11 x 11 means
again we are shaving off one pixel just like we did before .

So we are shaving off one pixel from the boundary and that reduces the image size And then we have MaxPool2 ,which 
also has the resolution in terms of the width and height.

So you can see whta's happening here as we go through these layers , we go deeper into the model and the image resolution is reducing at every step.we go from 26 down to 5 X 5.

Now from Maxpool layer ,we go to a fully connected network.

How do we connect all of these units to this fully connected layer?This is a vector of 50 units.how we go from macpool2(20X5X5)shape to fc1(1X50)shape?the answewr is we just 
linearize the output of the maxpool.

we have 10 units in the output layer because this is the most digits data set, so there are 10 categories in total.

The corresponding code for above description is avaiable as below link:


https://github.com/zhinoos-adibi/Deep-Learning-based-Computer_Vision/blob/main/DUDL_CNN_CNN4MNIST.ipynb

-------------------------------------------------------------------------------

**Some expression which we should know in deploymengt of neural networks in coding:**

**Refrence is :https://github.com/python-engineer/pytorchTutorial/blob/master/09_dataloader.py**


 gradient computation etc. not efficient for whole data set
 
 -> divide dataset into small batches

'''
**training loop**
for epoch in range(num_epochs):

    **loop over all batches**
    
    for i in range(total_batches):
    
        batch_x, batch_y = ...
'''

 epoch = one forward and backward pass of ALL training samples

 batch_size = number of training samples used in one forward/backward pass

 number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampels

 e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

 --> DataLoader can do the batch computation for us

 Implement a custom Dataset:

 inherit Dataset

 implement __init__ , __getitem__ , and __len__
 

some famous datasets are available in torchvision.datasets

 e.g. MNIST, Fashion-MNIST, CIFAR10, COCO
 
---------------------------------------------------------------------------------------------------------
**list of built-in pytorch transforms:**


Refrence:https://github.com/zhinoos-adibi/pytorchTutorial/blob/master/10_transformers.py


'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data 
during creation of the DataSet

complete list of built-in transforms: 

https://pytorch.org/docs/stable/torchvision/transforms.html

on image
---------
CenterCrop, Grayscale, Pad, RandomAffine

RandomCrop, RandomHorizontalFlip, RandomRotation

Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray

ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

-------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------


Good refrence for more information about kernel and pooling:

https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53


**More concept and descriptino from the above mentioned site:**

 The role of the ConvNet is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction.
 
 The objective of the Convolution Operation is to extract the high-level features such as edges, from the input image.
 
 **Other pictures from various resources which help to understanding the CNN's architecture.**
 
 ![image](https://user-images.githubusercontent.com/95547363/156652599-6e79ec5e-e87c-4f50-ad46-ad1be842d992.png)

------------------------------------------------------------------------------------------------


![image](https://user-images.githubusercontent.com/95547363/156652638-ab764211-2582-498a-b11f-75a84ee8b4a9.png)

-------------------------------------------------------------------------------------------------


![image](https://user-images.githubusercontent.com/95547363/156652746-99898502-b908-4f49-b862-7dcf2af23520.png)


----------------------------------------------------------------------------------------------------------
What is 2D Gaussians?

Brief Description. The Gaussian smoothing operator is a 2-D convolution operator that is used to `blur' images and remove detail and noise. In this sense it
is similar to the mean filter, but it uses a different kernel that represents the shape of a Gaussian (`bell-shaped') hump.


![image](https://user-images.githubusercontent.com/95547363/158394312-47fc2b8f-38e4-4c17-a014-0f609745267b.png)


<img width="799" alt="image" src="https://user-images.githubusercontent.com/95547363/158485160-a311ace6-9218-4526-8ae5-82cb9ceca2ab.png">


It is common that with increasing depth in the CNN architecture , you decrease the pixel resolution.

while simultaneously increasing the representational resolution or the number of kernels that we are learning.

The goal of Convolutional and Max players is to do image processing.

The idea is to transform the image into sets of feature representations where the features are learned through back propagation .

All of that decision making and categorization stuff happens in the fully connected layers at the end.

![image](https://user-images.githubusercontent.com/95547363/158486121-290c2652-ca2f-4f58-b64f-7feca1dca52f.png)

