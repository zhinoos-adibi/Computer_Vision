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

The benefit of having the stride and the spatial extent be the samr number is that produces
 no overlap and no missing pixels.So we're not losing anything and we're not having any overlap.
 
 It is also possible to have the stride smaller than the kernel , and that creates some ovelapping wondows that is less common.


<img width="595" alt="image" src="https://user-images.githubusercontent.com/95547363/154861775-a6cfe899-fe3a-461d-9d31-60c0d0ddffc0.png">



