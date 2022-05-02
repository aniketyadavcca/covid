# %%
"""
# Import Libraries
"""

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# %%
"""
# Clone & Explore dataset
"""

# %%
#clone the dataset from the github repository
!git clone https://github.com/aniketyadavcca/datasets.git

# %%
#set the path to the main dir

import os
main_dir="/content/datasets/Data"

#set the path to the train dir
train_dir=os.path.join(main_dir,'train')

#set the path to the test dir
test_dir=os.path.join(main_dir,'test')


#directory with the training covid images
train_covid_dir=os.path.join(train_dir,'COVID19' )

#directory with the training normal images
train_normal_dir=os.path.join(train_dir,'NORMAL' )

#directory with the testing covid images
test_covid_dir=os.path.join(test_dir, 'COVID19')


#directory with the testing normal images
test_normal_dir=os.path.join(test_dir, 'NORMAL')


# %%
#print the filenames
train_covid_names=os.listdir(train_covid_dir)
print(train_covid_names[:10])

train_normal_names=os.listdir(train_normal_dir)
print(train_normal_names[:10])


test_covid_names=os.listdir(test_covid_dir)
print(test_covid_names[:10])

test_normal_names=os.listdir(test_normal_dir)
print(test_normal_names[:10])

# %%
#print the total no of images present in each dir
print("Total number of images present in the training set:", len(train_covid_names+train_normal_names))

# %%
"""
# Data Visualization
"""

# %%
# plot a grid of 16 images (8 images of Covid19 and 8 images of Normal)
import matplotlib.image as mpimg

#set the number of columns and rows
rows = 4
cols = 4

#set the figure size
fig=plt.gcf()
fig.set_size_inches(12,12)
#get the filenames from the covid & normal dir of the train dataset
covid_pic=[os.path.join(train_covid_dir, filename)for filename in train_covid_names[0:8]]
normal_pic=[os.path.join(train_normal_dir, filename)for filename in train_normal_names[0:8]]

#print the list
print(covid_pic)
print(normal_pic)

#merge the covid and normal list

merged_list = covid_pic+normal_pic
for i , img_path in enumerate(merged_list):
  data =img_path.split('/',6)[6]
  sp=plt.subplot(rows,cols,i+1)
  sp.axis('Off')
  img = mpimg.imread(img_path)
  sp.set_title(data, fontsize=10)
  plt.imshow(img, cmap='gray')

plt.show()

# %%
"""
# Data Preprocessing & Augmentation

"""

# %%
# generate training,testing and validation batches 
dgen_train=ImageDataGenerator(rescale=1./255,
                              validation_split=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True)
dgen_validation=ImageDataGenerator(rescale=1./255)
dgen_test=ImageDataGenerator(rescale=1./255)

train_generator=dgen_train.flow_from_directory(train_dir,
                                               target_size=(150,150),
                                               subset='training',
                                               batch_size=32,
                                               class_mode='binary')

validation_generator=dgen_train.flow_from_directory(train_dir,
                                               target_size=(150,150),
                                               subset='validation',
                                               batch_size=32,
                                               class_mode='binary')
test_generator=dgen_test.flow_from_directory(test_dir,
                                             target_size=(150,150),
                                             batch_size=32,
                                             class_mode='binary')
                                            
                        

# %%
#get the class indices
train_generator.class_indices

# %%
#get the image shape
train_generator.image_shape

# %%
"""
# Build Convolutional Neural Network Model
"""

# %%
model=Sequential()
# add the convolutional layer
# filters, size of filters,padding,activation_function,input_shape

model.add(Conv2D(32,(5,5), padding='SAME',activation='relu', input_shape=(150,150,3)))

# pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# place a dropout layer
model.add(Dropout(0.5))

# add another convolutional layer
model.add(Conv2D(64,(5,5), padding='SAME', activation='relu'))

# pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# place a dropout layer
model.add(Dropout(0.5))

# Flatten layer
model.add(Flatten())

# add a dense layer : amount of nodes, activation
model.add(Dense(256, activation='relu'))

# place a dropout layer
# 0.5 drop out rate is recommended, half input nodes will be dropped at each update
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()


# %%
"""
# Compile & Train the Model
"""

# %%
#compile the model
model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# %%
#train the model
history=model.fit(train_generator,
                  epochs=30,
                  validation_data=validation_generator)

# %%
"""
# Performance Evaluation
"""

# %%
#get the keys of history object
history.history.keys()

# %%
#plot graph between training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Training and validation losses')
plt.xlabel('epoch')

# %%
#plot graph between training and validation accuarcy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training','Validation'])
plt.title('Training and validation accuracy')
plt.xlabel('epoch')


# %%
# get the test acuarcy and loss
test_loss, test_acc=model.evaluate(test_generator)
print('Test loss: {} test acc: {}' .format(test_loss, test_acc))

# %%
"""
# Prediction On New Data
"""

# %%
from google.colab import files
from keras.preprocessing import image
uploaded=files.upload()
for filename in uploaded.keys():
  img_path='/content/'+filename
  img=image.load_img(img_path, target_size=(150,150))
  images=image.img_to_array(img)
  images=np.expand_dims(images, axis=0)
  prediction=model.predict(images)
  # print(filename)
  plt.imshow(img)
  if prediction ==0:
    print('Covid detected')
  else:
    print('Report is normal')