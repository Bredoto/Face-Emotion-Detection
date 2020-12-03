## System Requirements
- Python3
- TensorFlow2
- Keras
- OpenCV
## Dataset Preparation
In this project, we use [Emotion Dataset from Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) as our dataset. This dataset have 27,462 images with 5 category [“Angry”, “Happy”, “Neutral”, “Sad”, “Surprise”]. The images are in grayscale with 48px width and 48px height. Before training, we need to preprocess our dataset. Usually we need to implement data augmentation to increase training dataset and normalization to prevent bias on the dataset.
```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
```

I implemented normalization, rotation, zoom, flip and shear on train dataset so we will have more images can fit into neural network. I only implemented normalization to validation dataset as we no need to increase the validation images.

### Neural Network Configuration
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 48, 48, 32)        320       
_________________________________________________________________
activation_1 (Activation)    (None, 48, 48, 32)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 48, 48, 32)        128       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 48, 48, 32)        9248      
_________________________________________________________________
activation_2 (Activation)    (None, 48, 48, 32)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 48, 48, 32)        128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 24, 24, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 24, 24, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
activation_3 (Activation)    (None, 24, 24, 64)        0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 24, 24, 64)        256       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 24, 24, 64)        36928     
_________________________________________________________________
activation_4 (Activation)    (None, 24, 24, 64)        0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 24, 24, 64)        256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 12, 12, 128)       73856     
_________________________________________________________________
activation_5 (Activation)    (None, 12, 12, 128)       0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 12, 12, 128)       512       
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 12, 12, 128)       147584    
_________________________________________________________________
activation_6 (Activation)    (None, 12, 12, 128)       0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 12, 12, 128)       512       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 128)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 6, 6, 128)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 6, 6, 256)         295168    
_________________________________________________________________
activation_7 (Activation)    (None, 6, 6, 256)         0         
_________________________________________________________________
batch_normalization_7 (Batch (None, 6, 6, 256)         1024      
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 6, 6, 256)         590080    
_________________________________________________________________
activation_8 (Activation)    (None, 6, 6, 256)         0         
_________________________________________________________________
batch_normalization_8 (Batch (None, 6, 6, 256)         1024      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 3, 3, 256)         0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 3, 3, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                147520    
_________________________________________________________________
activation_9 (Activation)    (None, 64)                0         
_________________________________________________________________
batch_normalization_9 (Batch (None, 64)                256       
_________________________________________________________________
dropout_5 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_________________________________________________________________
activation_10 (Activation)   (None, 64)                0         
_________________________________________________________________
batch_normalization_10 (Batc (None, 64)                256       
_________________________________________________________________
dropout_6 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 325       
_________________________________________________________________
activation_11 (Activation)   (None, 5)                 0         
=================================================================
Total params: 1,328,037
Trainable params: 1,325,861
Non-trainable params: 2,176
_________________________________________________________________
None
```
There are 2 main blocks I implemented in this neutral network, Convolutional Block and Fully-Connected Block. In each Convolutional Block, it consists of 2 3×3 kernal convolution, 2 RELU Activation Function, 2 Batch Normalization, 1 Batch Normalization, 1 Max Pooling and 1 Dropout. The batch normalization layer implemented to speed up the training speed up the training time and make the model more generalize. Besides, dropout layer is used to randomly shut down the nodes inside the neutral network to prevent overfitting. In my network, there are 1,328,037 parameters and 1,325,861 are trainable parameters.
```
Epoch 1/25
755/755 [==============================] - 97s 129ms/step - loss: 1.8598 - 
accuracy: 0.2375 - val_loss: 1.5425 - val_accuracy: 0.2944

Epoch 00001: val_loss improved from inf to 1.54247, saving model to 
EmotionDetectionModelv1.h5
Epoch 2/25
755/755 [==============================] - 94s 124ms/step - loss: 1.5762 - 
accuracy: 0.2844 - val_loss: 1.4709 - val_accuracy: 0.3080

Epoch 00002: val_loss improved from 1.54247 to 1.47092, saving model to 
EmotionDetectionModelv1.h5
Epoch 3/25
755/755 [==============================] - 92s 122ms/step - loss: 1.5462 - 
accuracy: 0.3031 - val_loss: 1.4547 - val_accuracy: 0.3379

Epoch 00003: val_loss improved from 1.47092 to 1.45472, saving model to 
EmotionDetectionModelv1.h5
Epoch 4/25
755/755 [==============================] - 93s 124ms/step - loss: 1.5091 - 
accuracy: 0.3312 - val_loss: 1.2539 - val_accuracy: 0.3850

Epoch 00004: val_loss improved from 1.45472 to 1.25390, saving model to 
EmotionDetectionModelv1.h5
Epoch 5/25
755/755 [==============================] - 93s 124ms/step - loss: 1.4461 - 
accuracy: 0.3689 - val_loss: 1.3463 - val_accuracy: 0.4583

Epoch 00005: val_loss did not improve from 1.25390
Epoch 6/25
755/755 [==============================] - 95s 125ms/step - loss: 1.3575 - 
accuracy: 0.4193 - val_loss: 1.0812 - val_accuracy: 0.4808

Epoch 00006: val_loss improved from 1.25390 to 1.08124, saving model to 
EmotionDetectionModelv1.h5
Epoch 7/25
755/755 [==============================] - 95s 126ms/step - loss: 1.2761 - 
accuracy: 0.4680 - val_loss: 1.6224 - val_accuracy: 0.4963

Epoch 00007: val_loss did not improve from 1.08124
Epoch 8/25
755/755 [==============================] - 95s 126ms/step - loss: 1.2253 - 
accuracy: 0.4969 - val_loss: 1.8146 - val_accuracy: 0.5121

Epoch 00008: val_loss did not improve from 1.08124
Epoch 9/25
755/755 [==============================] - 94s 124ms/step - loss: 1.1897 - 
accuracy: 0.5168 - val_loss: 0.9683 - val_accuracy: 0.5508

Epoch 00009: val_loss improved from 1.08124 to 0.96832, saving model to 
EmotionDetectionModelv1.h5
Epoch 10/25
755/755 [==============================] - 93s 124ms/step - loss: 1.1648 - 
accuracy: 0.5305 - val_loss: 1.9766 - val_accuracy: 0.5383

Epoch 00010: val_loss did not improve from 0.96832
Epoch 11/25
755/755 [==============================] - 93s 124ms/step - loss: 1.1331 - 
accuracy: 0.5432 - val_loss: 1.2761 - val_accuracy: 0.5319

Epoch 00011: val_loss did not improve from 0.96832
Epoch 12/25
755/755 [==============================] - 94s 124ms/step - loss: 1.1121 - 
accuracy: 0.5559 - val_loss: 1.6933 - val_accuracy: 0.5874
Restoring model weights from the end of the best epoch

Epoch 00012: val_loss did not improve from 0.96832

Epoch 00012: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
Epoch 00012: early stopping
```
I implemented Early Stopping function in this model. The training process stopped at 12nd epochs of training as the validation loss did not improve from 0.96832 and the learning reduced to 0.0002. The accuracy is around 55%. We can increase the accuracy by tuning the hyperparameters.

## Real-time Detection using OpenCV
OpenCV is needed to let us enable the usage of Web Camera/Video with Python. First, I use Haar Cascade Face Detection to detect the location of face. For faces detected in each frame, I convert it into grayscale, resize to 48×48, normalize it and pass into model to perform prediction. The result will display at the bottom of the bounding box.

![Real-time Face Emotion Detection](https://i2.wp.com/techyhans.com/wp-content/uploads/2020/11/result.gif?resize=585%2C585&ssl=1)

As you can see, even though the accuracy only 55%, the result is quite good.

## Conclusion
In this article, I covered the system flow of Emotion Detection from source of getting emotion dataset, dataset preprocessing, neural network configuration, and real-time detection using OpenCV. Feel free to contact me if you facing any problems. Thanks.

## Blog
[TechyHans - Han Sheng's Blog](https://techyhans.com)
