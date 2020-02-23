#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:35:30 2020

@author: edward
"""
import feather
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import ImageFile
from IPython.display import Image 
from tensorflow.keras.applications import nasnet
#from tensorflow.keras.applications.resnet_v2  import ResNet152V2
from tensorflow.keras.metrics import top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Input, Lambda, AveragePooling2D
from tensorflow import set_random_seed
import tensorflow as tf
import tensorflow.keras.backend as K
from cv_image_gen import Our_seq_generator
from copy import deepcopy


run_name = "ENET_train89"
seasons_to_train = [8,9]
train_seasons = ["SER_S" + str(i) for i in seasons_to_train]

# seq_id_use = feather.read_dataframe("curated_seq_ids.feather")

set_random_seed(2222)

# This is where our downloaded images and metadata live locally
train_metadata = pd.read_csv("data/train_metadata.csv")
train_labels = pd.read_csv("data/train_labels.csv", index_col="seq_id")
train_metadata = train_metadata.sort_values('file_name').groupby('seq_id').first()


## Training data
# train_metadata_use = train_metadata[train_metadata.index.isin(seq_id_use.seq_id)]
# train_labels_use = train_labels[train_labels.index.isin(seq_id_use.seq_id)]
# train_gen_df = train_labels_use.join(train_metadata_use.file_name.apply(lambda path: str(path)))
# IMAGE_DIR = Path("./data/images")
# train_gen_df['file_name'] = train_gen_df.apply(
#     lambda x: (IMAGE_DIR / x.file_name), axis=1
# )
# train_gen_df.file_name = train_gen_df.file_name.apply(lambda path: str(path))

train_metadata['season'] = train_metadata.index.map(lambda x: x.split('#')[0])
train_metadata_fortrain = deepcopy(train_metadata[train_metadata.season.isin(train_seasons)])

IMAGE_DIR = Path("./data/images")
train_metadata_fortrain['file_name'] = train_metadata_fortrain.apply(
    lambda x: (IMAGE_DIR / x.file_name), axis=1
)
train_metadata_fortrain.file_name = train_metadata_fortrain.file_name.apply(lambda path: str(path))
train_labels = train_labels[train_labels.index.isin(train_metadata_fortrain.index)]
train_gen_df = train_labels.join(train_metadata_fortrain.file_name)


label_columns = train_labels.columns.tolist()


## Val data season 10 ===============================================
train_metadata['season'] = train_metadata.index.map(lambda x: x.split('#')[0])
train_metadata = train_metadata[train_metadata.season.isin(["SER_S10"])]

IMAGE_DIR = Path("./data/images")
train_metadata['file_name'] = train_metadata.apply(
    lambda x: (IMAGE_DIR / x.file_name), axis=1
)

## 
train_metadata.file_name = train_metadata.file_name.apply(lambda path: str(path))
train_labels = train_labels[train_labels.index.isin(train_metadata.index)]
val_gen_df = train_labels.join(train_metadata.file_name)


ImageFile.LOAD_TRUNCATED_IMAGES = True
# This will be the input size to our model.
target_size = (384, 512)
batch_size = 32

to_drop_train = train_gen_df[(train_gen_df["empty"] == 1)].sample(frac=.7, random_state=123).index
train_gen_df = train_gen_df.drop(to_drop_train)


train_datagen = Our_seq_generator(target_size, train_gen_df.sample(frac=.2, random_state=123), batch_size, label_columns)
val_datagen = Our_seq_generator(target_size, val_gen_df.sample(frac=.05, random_state=123), batch_size, label_columns)


# train_gen_df['file_exists'] = train_gen_df.file_name.map(lambda x: os.path.isfile(x))


def get_transfer_model(model_to_transfer, num_classes, img_height, img_width, num_channels=3):
    inputs = Input(shape=(img_height, img_width, num_channels), name="inputs")
    transfer = model_to_transfer(include_top=False)
    
     # freeze layer weights in the transfer model to speed up training
    num = len(transfer.layers)
    i = 0
    for layer in transfer.layers:
        if i < num*.8:
            layer.trainable = False
        i = i + 1
        
    transfer_out = transfer(inputs)
    pooled = GlobalAveragePooling2D(name="pooling1")(transfer_out)
    outputs = Dense(num_classes, activation="sigmoid", name="classifer")(pooled)
    model = Model(inputs=inputs, outputs=outputs)
    return model


K.clear_session()

import efficientnet.tfkeras as efn

# EB5 = efn.EfficientNetB5(weights = "imagenet", )

model = get_transfer_model(
    model_to_transfer=efn.EfficientNetB5,
    num_classes=train_labels.shape[1],
    img_height=target_size[0],
    img_width=target_size[1],
)
model.summary()



metrics=[top_k_categorical_accuracy, categorical_crossentropy]
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics)


model_callback = tf.keras.callbacks.ModelCheckpoint("./inference/assets/check" + run_name + ".h5", monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

log_dir="logs/fit/" + run_name 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)


history = model.fit_generator(
    train_datagen,
    # steps_per_epoch=10,
    validation_data=val_datagen,
    # validation_steps=20,
    verbose = 1,
    epochs=10,
    workers=16,
    use_multiprocessing = True,
    callbacks=[tensorboard_callback, model_callback]
    )


model.save('./inference/assets/' + run_name + '.h5')
