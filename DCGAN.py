#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:53:41 2019

@author: feiqiwang
"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#For loading data
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import scipy
from glob import glob
from data_loader import DataLoader
import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import h5py
"""
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        'dataset/',
        target_size=(100, 100),
        batch_size=32,
        seed = 0)

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
"""

class DCGAN():
    def __init__(self,attemptN):
        # Input shape
        self.img_rows = 100
        self.img_cols = 100
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.dataset_name = "dataset"
        self.attempt = attemptN
        optimizer_dis = Adam(0.0001)
        optimizer_gen = Adam(0.0002, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_dis,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

        #Loading data
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

    def build_generator(self):

        model = Sequential()

        model.add(Dense(64 * 25 * 25, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((25, 25, 64)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=32, save_interval=20):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()
        
        # Rescale -1 to 1
        #X_train = X_train / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            #idx = np.random.randint(0, X_train.shape[0], batch_size)
            #imgs = X_train[idx]
            data_path = glob('./dataset/dataset/*')
            n_batches = int(len(path_A)/ batch_size)

            for i in range(n_batches - 1):
                
                imgs = self.data_loader.load_batch(i,batch_size)
            
                #imgs = np.expand_dims(imgs, axis=0)
                #print(imgs.shape)
                #print(imgs)
                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator (real classified as ones and generated as zeros)
            
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)


                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
                print ("epoch %d batch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, i, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.save_models(epoch)

    def save_imgs(self, epoch, round = 20):
        os.makedirs('./%s/Results/Epoch%d' % (self.attempt, epoch), exist_ok=True)
        for i in range(round):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_imgs = dcgan.generator.predict(noise)
                    # Rescale images 0 - 1
            gen_imgs = (0.5 * gen_imgs + 0.5)
            plt.imshow(gen_imgs[0, :,:,:])
            plt.axis('off')
            plt.savefig("./%s/Results/Epoch%d/image%d.png"%(self.attempt, epoch,i))
            plt.close()

    def save_models(self, epoch):
        os.makedirs('./%s/Models/Epoch%d' % (self.attempt, epoch), exist_ok=True)
        self.generator.save("./%s/Models/Epoch%d/generator.h5"% (self.attempt, epoch))
        self.discriminator.save("./%s/Models/Epoch%d/discriminator.h5"% (self.attempt, epoch))
        
if __name__ == '__main__':
    dcgan = DCGAN("Attempt0")
    dcgan.train(epochs=200, batch_size=32, save_interval=50)
    


