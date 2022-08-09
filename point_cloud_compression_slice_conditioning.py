# -*- coding: utf-8 -*-

"""
The code is taken from the tensorflow-compression github:
https://github.com/tensorflow/compression/blob/master/models/ms2020.py
and has been adapted to 3D point clouds compression.
"""

import argparse
import glob
import sys
import copy
import os
import tensorflow as tf
from src import pc_io
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

from src.partition import partition_pc
from src.processing import load_pc, pc_to_tf, process_x, create_dataset
from src.compression_utilities import pack_tensor, unpack_tensor, unpack_pc_bitstream, get_pc_header, po2po, compute_optimal_threshold
from src.evaluate import evaluate_pc
from absl import app
from absl.flags import argparse_flags
import tensorflow_compression as tfc
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.focal_loss import focal_loss
import functools
import itertools
import time
from tqdm.auto import tqdm




class Residual_Block(tf.keras.layers.Layer):
    """Residual transform used in the Analysis and Synthesis transform"""
    
    def __init__(self, num_filters, name):
        super().__init__(name=name)
        self.block1 = tf.keras.layers.Conv3D(
        num_filters/4, (3,3,3), padding='same',
        activation = 'relu')

        self.block2 = tf.keras.layers.Conv3D(
        num_filters/2, (3,3,3), padding='same',
        activation = 'relu')

        self.block3 = tf.keras.layers.Conv3D(
        num_filters/4, (1,1,1), padding='same',
        activation = 'relu')

        self.block4 = tf.keras.layers.Conv3D(
        num_filters/4, (3,3,3), padding='same',
        activation = 'relu')

        self.block5 = tf.keras.layers.Conv3D(
        num_filters/2, (1,1,1), padding='same',
        activation = 'relu')

        self.concat = tf.keras.layers.Concatenate()
        self.add = tf.keras.layers.Add()


    def call(self, x):
        y1 = self.block1(x)
        y1 = self.block2(y1)

        y2 = self.block3(x)
        y2 = self.block4(y2)
        y2 = self.block5(y2)

        concat = self.concat([y1,y2])
        output = self.add([x,concat])
        return output




class AnalysisTransform(tf.keras.layers.Layer):
    """Analysis transform used to turn the input into its latent representation"""
    
    def __init__(self, num_filters, latent_depth):
        super().__init__(name='analysis')
        
            
        self.conv = tf.keras.layers.Conv3D(
        num_filters, (9,9,9), strides = (2,2,2), padding='same',
        activation = 'relu')
        
        self.conv_int = tf.keras.layers.Conv3D(
        num_filters, (5,5,5), strides = (2,2,2), padding='same',
        activation = 'relu')
        
        self.convout = tf.keras.layers.Conv3D(
        latent_depth, (5,5,5), strides = (2,2,2), padding='same',
        activation = 'linear')

        self.res_block1 = Residual_Block(num_filters, name='block_1')
        self.res_block2 = Residual_Block(num_filters, name='block_2')
        
            
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.res_block1(x)
        x = self.conv_int(x)
        x = self.res_block2(x)
        x = self.convout(x)
        return x

            
class SynthesisTransform(tf.keras.layers.Layer):
    """Analysis transform used to turn the input into its latent representation"""
    
    def __init__(self, num_filters):
        super().__init__(name='synthesis')
            
        self.conv1 = tf.keras.layers.Conv3DTranspose(
        num_filters, (5,5,5), strides=(2,2,2), padding='same',
        activation='relu')
        
        self.conv2 = tf.keras.layers.Conv3DTranspose(
        num_filters, (5,5,5), strides=(2,2,2), padding='same',
        activation='relu')
        
        self.conv3 = tf.keras.layers.Conv3DTranspose(
        1, (9,9,9), strides=(2,2,2), padding='same',
        activation='sigmoid')
        
        self.block1 = Residual_Block(num_filters, name='block_3')
        self.block2 = Residual_Block(num_filters, name='block_4')
       
    def call(self, inputs):
      x = self.conv1(inputs)
      x = self.block1(x)
      x = self.conv2(x)
      x = self.block2(x)
      x = self.conv3(x)
      return x

class HyperAnalysisTransform(tf.keras.layers.Layer):
    """Analysis transform for the entropy model's parameters."""
    
    def __init__(self, num_filters, hyperprior_depth):
        super().__init__(name='hyper_analysis')
        
        self.conv1 = tf.keras.layers.Conv3D(
        num_filters, (3,3,3), padding='same',
        activation='relu')
        
        self.conv2 = tf.keras.layers.Conv3D(
        num_filters, (3,3,3), strides = (2,2,2), padding='same',
        activation='relu')
        
        self.conv3 = tf.keras.layers.Conv3D(
        hyperprior_depth, (1,1,1), padding='same',
        activation='linear', use_bias=False)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """Synthesis transform for the entropy model's parameters."""
    
    def __init__(self, num_filters):
        super().__init__(name='hyper_synthesis')
        
        self.convt1 = tf.keras.layers.Conv3DTranspose(
        num_filters, (1,1,1), padding='same',
        activation='relu')
        
        self.convt2 = tf.keras.layers.Conv3DTranspose(
        num_filters, (3,3,3), strides=(2,2,2), padding='same',
        activation='relu')
        
        self.convt3 = tf.keras.layers.Conv3DTranspose(
        num_filters, (3,3,3), padding='same',
        activation='relu')

    def call(self, input):
        x = self.convt1(input)
        x = self.convt2(x)
        x = self.convt3(x)
        return x

class SliceTransform(tf.keras.layers.Layer):
  """Transform for channel-conditional params and latent residual prediction."""

  def __init__(self, num_filters, latent_depth, num_slices):
    super().__init__(name='slice_transform')

    # Note that the number of channels in the output tensor must match the
    # size of the corresponding slice. If we have 10 slices and a bottleneck
    # with 320 channels, the output is 320 / 10 = 32 channels.
    slice_depth = latent_depth // num_slices
    if slice_depth * num_slices != latent_depth:
      raise ValueError('Slices do not evenly divide latent depth (%d / %d)' % (
          latent_depth, num_slices))

    self.transform = tf.keras.Sequential([
        tf.keras.layers.Conv3D(
        num_filters, (3,3,3), padding='same',
        activation='relu'),
        tf.keras.layers.Conv3D(
        num_filters, (3,3,3), padding='same',
        activation='relu'),
        tf.keras.layers.Conv3D(
        slice_depth, (3,3,3), padding='same',
        activation='linear'),
    ])

  def call(self, tensor):
    return self.transform(tensor)


class CompressionModel(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, alpha, num_filters, 
               latent_depth, hyperprior_depth,
               num_slices, max_support_slices,
               num_scales, scale_min, scale_max):
    super().__init__()
    self.lmbda = lmbda
    self.alpha = alpha
    self.latent_depth = latent_depth
    self.num_filters = num_filters
    self.num_scales = num_scales
    self.num_slices = num_slices
    self.max_support_slices = max_support_slices
    
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    
    self.analysis_transform = AnalysisTransform(num_filters, latent_depth)
    self.synthesis_transform = SynthesisTransform(num_filters)
    
    self.hyper_analysis_transform = HyperAnalysisTransform(num_filters, hyperprior_depth)
    self.hyper_synthesis_mean_transform = HyperSynthesisTransform(num_filters)
    self.hyper_synthesis_scale_transform = HyperSynthesisTransform(num_filters)
    
    self.cc_mean_transforms = [
        SliceTransform(num_filters, latent_depth, num_slices) for _ in range(num_slices)]
    self.cc_scale_transforms = [
        SliceTransform(num_filters, latent_depth, num_slices) for _ in range(num_slices)]
    self.lrp_transforms = [
        SliceTransform(num_filters, latent_depth, num_slices) for _ in range(num_slices)]
        
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=[hyperprior_depth])
    
    self.build((None, None, None, None, 1))
    
    # The call signature of decompress() depends on the number of slices, so we
    # need to compile the function dynamically.
    self.decompress = tf.function(
        input_signature=3 * [tf.TensorSpec(shape=(3,), dtype=tf.int32)] +
        (num_slices + 1) * [tf.TensorSpec(shape=(1,), dtype=tf.string)]
    )(self.decompress)

  def call(self, x, training):
    """Computes rate and distortion losses."""
    
    geo_x = x[:, :, :, :, 0]
        
    num_voxels = tf.cast(tf.size(geo_x), tf.float32)
    num_occupied_voxels = tf.reduce_sum(geo_x)
    
    # Build the encoder (analysis) half of the hierarchical autoencoder.
    y = self.analysis_transform(x)
    y_shape = tf.shape(y)[1:-1]
    
    # Build the encoder (analysis) half of the hyper-prior.
    z = self.hyper_analysis_transform(y)

    # Build the entropy model for the hyperprior (z).
    em_z = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=4, compression=False)

    # When training, z_bpp is based on the noisy version of z (z_tilde).
    _, z_bits = em_z(z, training=training)
    z_bpp = tf.reduce_sum(z_bits) / num_occupied_voxels

    # Use rounding (instead of uniform noise) to modify z before passing it
    # to the hyper-synthesis transforms. Note that quantize() overrides the
    # gradient to create a straight-through estimator.
    z_hat = em_z.quantize(z)

    # Build the decoder (synthesis) half of the hyper-prior.
    latent_scales = self.hyper_synthesis_scale_transform(z_hat)
    latent_means = self.hyper_synthesis_mean_transform(z_hat)

    # Build a conditional entropy model for the slices.
    em_y = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
        coding_rank=4, compression=False)

    # En/Decode each slice conditioned on hyperprior and previous slices.
    y_slices = tf.split(y, self.num_slices, axis=-1)
    y_hat_slices = []
    y_bpps = []
    for slice_index, y_slice in enumerate(y_slices):
        
      # Model may condition on only a subset of previous slices.
      support_slices = (y_hat_slices if self.max_support_slices < 0 else
                        y_hat_slices[:self.max_support_slices])

      # Predict mu and sigma for the current slice.
      mean_support = tf.concat([latent_means] + support_slices, axis=-1)
      mu = self.cc_mean_transforms[slice_index](mean_support)
      mu = mu[:, :y_shape[0], :y_shape[1], :]

      # Note that in this implementation, `sigma` represents scale indices,
      # not actual scale values.
      scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
      sigma = self.cc_scale_transforms[slice_index](scale_support)
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]

      _, slice_bits = em_y(y_slice, sigma, loc=mu, training=training)
      slice_bpp = tf.reduce_sum(slice_bits) / num_occupied_voxels
      y_bpps.append(slice_bpp)

      # For the synthesis transform, use rounding. Note that quantize()
      # overrides the gradient to create a straight-through estimator.
      y_hat_slice = em_y.quantize(y_slice, sigma, loc=mu)

      # Add latent residual prediction (LRP).
      lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
      lrp = self.lrp_transforms[slice_index](lrp_support)
      lrp = 0.5 * tf.math.tanh(lrp)
      y_hat_slice += lrp

      y_hat_slices.append(y_hat_slice)

    # Merge slices and generate the bloc reconstruction.
    y_hat = tf.concat(y_hat_slices, axis=-1)
    x_hat = self.synthesis_transform(y_hat)

    # Total bpp is sum of bpp from hyperprior and all slices.
    total_bpp = tf.add_n(y_bpps + [z_bpp])
    y_bpp = tf.add_n(y_bpps)
    z_bpp = tf.add_n([z_bpp])
    
    # Compute the focal loss and/or color loss across pixels.
    # Don't clip or round pixel values while training.
    fcl = focal_loss(x, x_hat, gamma = 2, alpha = self.alpha) / num_voxels
      
    # Calculate and return the rate-distortion loss.
    loss = total_bpp + self.lmbda * fcl

    return loss, y_bpp, z_bpp, total_bpp, fcl
    

  def train_step(self, x):
    with tf.GradientTape(persistent=True) as tape:
        # Compute the loss under the gradient tape to prepare for 
        # gradient computation.
        loss, bpp1, bpp2, bpp, fcl = self(x, training=True)
    
    # Gather the trainable variables.
    variables = self.trainable_variables
    prior_var = self.hyperprior.trainable_variables
    
    # Compute the gradients w.r.t the trainable variables.
    gradients = tape.gradient(loss, variables)
    grad_prior = tape.gradient(loss, prior_var)
    
    # Apply the gradient to update the variables.
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.optimizer.apply_gradients(zip(grad_prior, prior_var))
    
    # Update the values displayed during training.
    self.loss.update_state(loss)
    self.bpp1.update_state(bpp1)
    self.bpp2.update_state(bpp2)
    self.bpp.update_state(bpp)
    self.fcl.update_state(self.lmbda*fcl)
    
    # Return the appropriate informations to be displayed on screen.
    return {m.name: m.result() for m in [self.loss, self.bpp1, self.bpp2, self.bpp, self.fcl]}

  def test_step(self, x):
    loss, bpp1, bpp2, bpp, fcl = self(x, training=False)
    
    # Update the values displayed during validation.
    self.loss.update_state(loss)
    self.bpp1.update_state(bpp1)
    self.bpp2.update_state(bpp2)
    self.bpp.update_state(bpp)
    self.fcl.update_state(self.lmbda*fcl)
    
    # Return the appropriate informations to be displayed on screen.
    return {m.name: m.result() for m in [self.loss, self.bpp1, self.bpp2, self.bpp, self.fcl]}

  def predict_step(self, x):
    raise NotImplementedError('Prediction API is not supported.')

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name='loss')
    self.bpp1 = tf.keras.metrics.Mean(name='direct_bpp')
    self.bpp2 = tf.keras.metrics.Mean(name='side_bpp')
    self.bpp = tf.keras.metrics.Mean(name='bpov')
    self.fcl = tf.keras.metrics.Mean(name='focal_loss')

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    
    # After training, fix range coding tables.
    self.em_z = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=4, compression=True)
        
    self.em_y = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
        coding_rank=4, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
  ])
  def compress(self, x):
    """Compresses a block."""
    
    # Add batch dimension and cast to float.
    x = tf.expand_dims(x, 0)

    y_strings = []
    x_shape = tf.shape(x)[1:-1]

    # Build the encoder (analysis) half of the hierarchical autoencoder.
    y = self.analysis_transform(x)
    y_shape = tf.shape(y)[1:-1]
    
    # Build the encoder (analysis) half of the hyper-prior.
    z = self.hyper_analysis_transform(y)
    z_shape = tf.shape(z)[1:-1]
    
    # Compress the output of the Hyper-Analysis to pass it
    # in the bistream.
    z_string = self.em_z.compress(z)
    z_hat = self.em_z.decompress(z_string, z_shape)

    # Build the decoder (synthesis) half of the hyper-prior.
    latent_scales = self.hyper_synthesis_scale_transform(z_hat)
    latent_means = self.hyper_synthesis_mean_transform(z_hat)

    # En/Decode each slice conditioned on hyperprior and previous slices.
    y_slices = tf.split(y, self.num_slices, axis=-1)
    y_hat_slices = []
    for slice_index, y_slice in enumerate(y_slices):
        
    # Model may condition on only a subset of previous slices.
      support_slices = (y_hat_slices if self.max_support_slices < 0 else
                        y_hat_slices[:self.max_support_slices])

      # Predict mu and sigma for the current slice.
      mean_support = tf.concat([latent_means] + support_slices, axis=-1)
      mu = self.cc_mean_transforms[slice_index](mean_support)
      mu = mu[:, :y_shape[0], :y_shape[1], :]

      # Note that in this implementation, `sigma` represents scale indices,
      # not actual scale values.
      scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
      sigma = self.cc_scale_transforms[slice_index](scale_support)
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]

      slice_string = self.em_y.compress(y_slice, sigma, mu)
      y_strings.append(slice_string)
      y_hat_slice = self.em_y.decompress(slice_string, sigma, mu)

      # Add latent residual prediction (LRP).
      lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
      lrp = self.lrp_transforms[slice_index](lrp_support)
      lrp = 0.5 * tf.math.tanh(lrp)
      y_hat_slice += lrp

      y_hat_slices.append(y_hat_slice)

    return (x_shape, y_shape, z_shape, z_string) + tuple(y_strings)

  def decompress(self, x_shape, y_shape, z_shape, z_string, *y_strings):
    """Decompresses a block."""
    
    assert len(y_strings) == self.num_slices
    
    # Recover the entropy parameters.
    z_hat = self.em_z.decompress(z_string, z_shape)

    # Build the decoder (synthesis) half of the hierarchical autoencoder.
    latent_scales = self.hyper_synthesis_scale_transform(z_hat)
    latent_means = self.hyper_synthesis_mean_transform(z_hat)

    # En/Decode each slice conditioned on hyperprior and previous slices.
    y_hat_slices = []
    for slice_index, y_string in enumerate(y_strings):
      # Model may condition on only a subset of previous slices.
      support_slices = (y_hat_slices if self.max_support_slices < 0 else
                        y_hat_slices[:self.max_support_slices])

      # Predict mu and sigma for the current slice.
      mean_support = tf.concat([latent_means] + support_slices, axis=-1)
      mu = self.cc_mean_transforms[slice_index](mean_support)
      mu = mu[:, :y_shape[0], :y_shape[1], :]

      # Note that in this implementation, `sigma` represents scale indices,
      # not actual scale values.
      scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
      sigma = self.cc_scale_transforms[slice_index](scale_support)
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]

      y_hat_slice = self.em_y.decompress(y_string, sigma, loc=mu)

      # Add latent residual prediction (LRP).
      lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
      lrp = self.lrp_transforms[slice_index](lrp_support)
      lrp = 0.5 * tf.math.tanh(lrp)
      y_hat_slice += lrp

      y_hat_slices.append(y_hat_slice)

    # Merge slices and generate the image reconstruction.
    y_hat = tf.concat(y_hat_slices, axis=-1)
    x_hat = self.synthesis_transform(y_hat)
    
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    
    # The transformation of x_hat into an occupancy map is done at a 
    # later stage to allow for the computation of the adaptive threshold.
    return x_hat


def train(args):
    """Instantiates and trains the model."""
        
    if args.check_numerics:
        tf.debugging.enable_check_numerics()
            
    # Load the raw point cloud blocks.
    points_train, points_val, dense_tensor_shape = load_pc(args)
        
    # Transform the raw points into a training and validation dataset.
    train_dataset = create_dataset(points_train, dense_tensor_shape,
                                    args, repeat=True)
    validation_dataset = create_dataset(points_val, dense_tensor_shape,
                                         args, repeat=False)
        
    # Instanciate the model.
    model = CompressionModel(args.lmbda, args.alpha, args.num_filters,
                             args.latent_depth, args.hyperprior_depth,
                             args.num_slices, args.max_support_slices,
                             args.num_scales, args.scale_min,
                             args.scale_max)
        
    # Compile the model.
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        )
            
    # Load previously trained model to speed up training.
    if args.checkpoint_path is not None:
        model.load_weights(os.path.join(args.model_path, args.checkpoint_path))
            
    # Get a summary of the model and prepare the paths for the logs.    
    model.summary()
    logdir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    file_path = os.path.join(args.model_path, args.experiment + '/')
        
    # Train the model.
    model.fit(
    train_dataset.prefetch(tf.data.AUTOTUNE),
    epochs=args.epochs,
    steps_per_epoch=args.steps_per_epoch,
    validation_data=validation_dataset.cache(),
    validation_freq=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
            patience=20,
            restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1, update_freq='epoch'),
        tf.keras.callbacks.experimental.BackupAndRestore(args.model_path),
        tf.keras.callbacks.ModelCheckpoint(filepath=file_path, save_weights_only=True,
                                             save_best_only=True)
         ],
    verbose=int(args.verbose),
     )
        
    # Save the model after training.
    model.save(os.path.join(args.model_path, args.experiment))


def compress(args):
    """Compresses a block."""
    
    # Load the model.
    model = tf.keras.models.load_model(os.path.join(args.model_path, args.experiment))
    
    # Get the files to be compressed.
    files = pc_io.get_files(args.input_glob)
    assert len(files) > 0, 'No input files found'

    # Load the blocks from the files.
    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(
        args.resolution, args.channels_last)
    #points = pc_io.load_points(files[:args.input_length], p_min, p_max)

    total_blocks = 0
    tqdm_handle = tqdm(files)

    print('Starting compression...')
    start = time.time()

    for pc_filepath in tqdm_handle:

        pc_blocks = partition_pc(pc_filepath, args.resolution, 0, True)
        pc_file = os.path.split(pc_filepath)[-1]

        pc_bitstream = get_pc_header(args.resolution, args.adaptive)

        for i, pc_block in enumerate(pc_blocks):

            pc_block_geo = pc_block[0][:,:3]
            pc_block_pos = pc_block[1]
            
            # Process the input for the neural network.
            tensor = pc_to_tf(pc_block_geo, dense_tensor_shape, args.channels_last)
            tensor = process_x(tensor, dense_tensor_shape)

            # Compress the input.
            tensors = model.compress(tensor)
            
            # Calculate the adaptive threshold.
            if args.adaptive:
                best_threshold = compute_optimal_threshold(model, tensors, pc_block_geo, delta_t = 0.01, breakpt = 150, verbose = 0)
            else:
                best_threshold = tf.constant(0.5)
                
            # Write a binary file with the shape information and the compressed string.
            block_bitstream = pack_tensor(best_threshold, pc_block_pos, *tensors, bytes_length = 2, adaptive = args.adaptive)

            pc_bitstream += block_bitstream

            tqdm_handle.set_description(f"Compressed {i+1}/{len(pc_blocks)} blocks of {pc_file}")
            

        if not os.path.isdir(os.path.join(args.output_dir, args.experiment)):
                os.mkdir(os.path.join(args.output_dir, args.experiment))

        with open(os.path.join(args.output_dir, args.experiment, pc_file[:-4] + '.bin'), 'wb') as f:
            f.write(pc_bitstream)
            
    print(f'Done. Total compression time: {time.time() - start}s')

def decompress(args):
    """Decompresses previously compressed blocks."""
    
    # Load the model and determine the dtypes of tensors required to decompress.
    model = tf.keras.models.load_model(os.path.join(args.model_path, args.experiment))
        
    # Load the .bin files to decompress.
    #files = pc_io.get_files(args.input_glob)[:args.input_length]
    files = pc_io.get_files(os.path.join(args.input_glob, args.experiment))

    assert len(files) > 0, 'No input files found'
    
    # Prepare the original files for debugging if requested.
    if args.debug:
        assert args.ori_input_glob is not None, 'Please specify the path to the uncompressed blocks for debugging'
        original = pc_io.get_files(args.ori_input_glob)
        assert len(original) > 0, 'No original files found'
        
        # p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(
        #     args.resolution, args.channels_last)
        # points = pc_io.load_points(original, p_min, p_max)
    

    print('Starting decompression...')
    tqdm_handle = tqdm(enumerate(files), total=len(files))
    start = time.time()
    for n, bin_filepath in tqdm_handle:

        if args.debug:
            original_blocks = partition_pc(original[n], args.resolution, 0, True)
        
        # Read the shape information and compressed string from the binary file,
        # and decompress the blocks using the model.
        with open(bin_filepath, 'rb') as f:
            pc_bitstream = f.read()

        bin_file = os.path.split(bin_filepath)[-1]
        resolution, adaptive, block_bitstreams = unpack_pc_bitstream(pc_bitstream)

        pa = np.zeros([0,3])

        for i, block_bitstream in enumerate(block_bitstreams):

            z_string, y_strings, threshold, block_position = unpack_tensor(block_bitstream, bytes_length = 2,
                                                   adaptive = adaptive)

            x_size = resolution
            y_size = int(x_size / 8)
            z_size = int(y_size / 2)

            x_shape = tf.constant([x_size, x_size, x_size], dtype=tf.int32)
            y_shape = tf.constant([y_size, y_size, y_size], dtype=tf.int32)
            z_shape = tf.constant([z_size, z_size, z_size], dtype=tf.int32)

            x_hat = model.decompress(x_shape, y_shape, z_shape, z_string, *y_strings).numpy()
            
            # Transform the decompressed tensor into an occupancy map
            # according to threshold.
            block_pa = np.argwhere(x_hat[...,0] > threshold).astype('float32')
            block_pa += np.asarray(block_position) * resolution

            if pa.shape[0] == 0:
                pa = block_pa
            else:
                pa = np.concatenate([pa, block_pa], axis=0)
        
            # Compressing/decompressing a file using a GPU
            # often induces errors that propagate and ruin the reconstruction.
            # It is strongly recommended to run the compression/decompression
            # on CPU for this reason. The following section might however
            # help using a GPU by retrying the decompression until the
            # reconstruction is done correctly. It is not guarenteed that
            # it will happen.
            # More information at:
            # J. Ballé, N. Johnston, D. Minnen,
            # "Integer Networks for data compression with latent-variable models"
            # https://openreview.net/pdf?id=S1zz2i0cY7
            if args.debug:
                failed = True
                max_retries = 200
                
                # Check that the correct files are being compared.
                assert original_blocks[i][1] == block_position, 'The original file is not the same as the decompressed one'
                retry = 0
                
                # Redo decompression as long as it fails and max_retries
                # is not reached.
                while failed and retry <= max_retries:
                    if len(pa) == 0:
                        break
                    
                    # Compute the D1 metric to spot decompression issues.
                    mse = po2po(original_blocks[i][:,:3], pa)
                    print('Mse for block ' + str(i) + ': ' + str(mse))
                    
                    # Depending on the PC and the target bpp, different
                    # values of mse should be experimented there.
                    if mse <= 50:
                        failed = False
                    else:
                        retry += 1
                        print(f'Decompression of block from position {block_position} from point cloud {original[n]} failed, retrying. Attempt number ' + str(retry))
                        x_hat = model.decompress(*tensors).numpy()
                        pa = np.argwhere(x_hat[...,0] > threshold).astype('float32')
                     
            tqdm_handle.set_description(f"Decompressed {i+1}/{len(block_bitstreams)} blocks of {bin_file}")

        if not os.path.isdir(os.path.join(args.output_dir, args.experiment)):
            os.mkdir(os.path.join(args.output_dir, args.experiment))
        
        # Write the reconstructed block in a PLY file.
        pc_io.write_df(os.path.join(args.output_dir, args.experiment, os.path.split(bin_filepath)[-1][:-4] + '.ply'), pc_io.pa_to_df(pa))
            
            
        #if (i+1) % 50 == 0:
        #    print(f'Decompressed {i+1} files out of {len(files)}')
    print(f'Done. Total decompression time: {time.time() - start}s')


def parse_args(argv, mode='compress'):
        """Parses command line arguments."""
        parser = argparse_flags.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # High-level options.
        parser.add_argument(
            '--model_path', default='PCC',
            help='Path where to save/load the trained model and the results.')
        parser.add_argument(
            '--experiment', default='Point_cloud_compression',
            help='Name of the folder that will contain the results of training and evaluation')
        subparsers = parser.add_subparsers(
            title='commands', dest='command',
            help='What to do: "train" loads training data and trains (or continues '
            'to train) a new model. "compress" reads a voxel file and writes a compressed binary file. "decompress" '
            'reads a binary file and reconstructs the image (in PLY format). '
            'input and output filenames need to be provided for the latter '
            'two options. "evaluate" runs the mpeg-pcc error computation software '
            'and evaluates the rate distortion of the reconstructed PC. '
            ' Invoke "<command> -h" for more information.')

        # 'train' subcommand.
        train_cmd = subparsers.add_parser(
            'train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Trains (or continues to train) a new model. Note that this '
                    'model trains on a continuous stream of blocks drawn from '
                    'the training dataset. An epoch is always defined as '
                    'the same number of batches given by --steps_per_epoch. '
                    'The purpose of validation is mostly to evaluate the '
                    'rate-distortion performance of the model using actual '
                    'quantization rather than the differentiable proxy loss. '
                    )
        train_cmd.add_argument(
            '--lambda', type=float, default=0.01, dest='lmbda',
            help='Lambda for rate-distortion tradeoff.')
        train_cmd.add_argument(
            '--alpha', type=float, default=0.6, dest='alpha',
            help='Alpha parameter of the focal loss.')
        train_cmd.add_argument(
            '--train_glob', type=str, default=None,
            help='Glob pattern identifying training data. This pattern must '
                'expand to a list of RGB blocks in PLY format.')
        train_cmd.add_argument(
            '--num_filters', type=int, default=64,
            help='Number of filters per layer.')
        train_cmd.add_argument(
            '--latent_depth', type=int, default=160,
            help='Number of filters of the last layer of the analysis transform.')
        train_cmd.add_argument(
            '--hyperprior_depth', type=int, default=64,
            help='Number of filters of the last layer of the hyper-analysis '
                'transform.')
        train_cmd.add_argument(
            '--num_slices', type=int, default=10,
            help='Number of channel slices for conditional entropy modeling.')
        train_cmd.add_argument(
            '--max_support_slices', type=int, default=-1,
            help='Maximum number of preceding slices to condition the current slice '
                'on.')
        train_cmd.add_argument(
            '--num_scales', type=int, default=64,
            help='Number of Gaussian scales to prepare range coding tables for.')
        train_cmd.add_argument(
            '--scale_min', type=float, default=.11,
            help='Minimum value of standard deviation of Gaussians.')
        train_cmd.add_argument(
            '--scale_max', type=float, default=256.,
            help='Maximum value of standard deviation of Gaussians.')
        train_cmd.add_argument(
            '--resolution',
            type=int, help='Size of the blocks.', default=64)
        train_cmd.add_argument(
            '--checkpoint_path', default=None,
            help='Path to the checkpoint that will be used to resume '
            'training. Leave as None to start a fresh training.')
        train_cmd.add_argument(
            '--num_data', type=int, default=None,
            help='Number of total data we want to use (-1: use all data).')
        train_cmd.add_argument(
            '--channels_last', default=True,
            help='Use channels last instead of channels first.')
        train_cmd.add_argument(
            '--num_val', type=int, default=640,
            help='Number of validation data we want to use')
        train_cmd.add_argument(
            '--batch_size', type=int, default=16,
            help='Batch size for training and validation.')
        train_cmd.add_argument(
            '--epochs', type=int, default=1000,
            help='Train up at most to this number of epochs. (One epoch is here defined as '
                'the number of steps given by --steps_per_epoch, not iterations '
                'over the full training dataset.)')
        train_cmd.add_argument(
            '--steps_per_epoch', type=int, default=1000,
            help='Perform validation and produce logs after this many batches.')
        train_cmd.add_argument(
            '--preprocess_threads', type=int, default=16,
            help='Number of CPU threads to use for parallel decoding of training '
                'blocks.')
        train_cmd.add_argument(
            '--verbose', '-V', action='store_true',
            help='Report progress and metrics when training or compressing.')
        train_cmd.add_argument(
            '--check_numerics', action='store_true',
            help='Enable TF support for catching NaN and Inf in tensors.')

        # 'compress' subcommand.
        compress_cmd = subparsers.add_parser(
            'compress',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Reads a PLY file, compresses it, and writes a BIN file.')
        compress_cmd.add_argument(
                '--adaptive', action='store_true',
                help='Use adaptive thresholding.')

        # 'decompress' subcommand.
        decompress_cmd = subparsers.add_parser(
            'decompress',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Reads a .bin file, reconstructs the block,'
                        ' and writes back a PLY file.')
        decompress_cmd.add_argument(
            '--ori_dir',
            help='Directory containing the original unpartitioned point clouds.')
        decompress_cmd.add_argument(
            '--reconstructed_dir',
            help='Directory where the reconstructed point clouds will be saved.')
        decompress_cmd.add_argument(
            '--debug', action='store_true',
            help='Enable debug mode to decompress on GPU.')
        decompress_cmd.add_argument(
            '--ori_input_glob',
            help='Directory containing the original uncompressed blocks. For debugging GPU decompression only')




    # Arguments for both 'compress' and 'decompress'.
        for cmd, ext in ((compress_cmd, '.bin'), (decompress_cmd, '.ply')):
            cmd.add_argument(
                '--input_glob',
                help='Input directory.')
            cmd.add_argument(
                '--input_length', type=int, default=None,
                help='Number of files to process.')
            cmd.add_argument(
                '--resolution',
                type=int, help='Size of the blocks.', default=128)
            cmd.add_argument(
                '--output_dir', nargs='?',
                help='Output directory.')
            cmd.add_argument(
                '--channels_last', default=True,
                help='Use channels last instead of channels first.')
            cmd.add_argument(
                '--gpu', action='store_true',
                help='Use the GPU if available to compress/decompress. '
                'It is recommended to do the compression/decompression '
                ' on cpu.')
            
        # 'evaluate' subcommand
        evaluate_cmd = subparsers.add_parser(
            'evaluate',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Evaluates the reconstructed point clouds using '
            'the mpeg error computation software. Saves the '
            'results as a csv files, detailing the bpp, '
            'D1 and D2 metrics among other characteristics '
            'of the point cloud')
            
        evaluate_cmd.add_argument(
            '--ori_dir',
            help='Input directory with original (not compressed) point clouds.')
        evaluate_cmd.add_argument(
            '--dec_dir',
            help='Parent directory with reconstructed point clouds. the argument "experiment" '
            'is automatically added as the child directory.')
        evaluate_cmd.add_argument(
            '--nor_dir',
            help='Input directory with original point clouds with normal attributes.')
        evaluate_cmd.add_argument(
            '--bin_dir',
            help='Input directory with compressed point cloud binaries.')
        evaluate_cmd.add_argument(
            '--output_dir',
            help='Directory where to save the results of the evaluation.')
        evaluate_cmd.add_argument(
            '--pc_error_path',
            help='Path to the binary of MPEG evaluation metric software.')
        evaluate_cmd.add_argument(
                '--resolution',
                type=int, help='Resolution of the evaluated point clouds.', default=1023)

      
        args = parser.parse_args(argv[1:])
        if args.command is None:
            parser.print_usage()
            sys.exit(2)
        return args


def main(args):
    # Invoke subcommand.
    if args.command == 'train':
        train(args)
        
    elif args.command == 'compress':
        if not args.output_dir:
            args.output_file = args.input_glob + '.bin'
        if not args.gpu:
            with tf.device('/cpu:0'):
                compress(args)
        else:
            compress(args)
            
    elif args.command == 'decompress':
        if not args.output_dir:
            args.output_file = args.input_glob + '.ply'
        if not args.gpu:
            with tf.device('/cpu:0'):
                decompress(args)
        else:
            decompress(args)
            
        # assert args.resolution > 0, 'block_size must be positive'
        # args.ori_dir = os.path.normpath(args.ori_dir)
        # assert os.path.exists(args.ori_dir), 'Input directory with original (not compressed) point clouds not found'

        # ori_files = sorted([f for f in os.listdir(args.ori_dir) if '.ply' in f])
        # print(f'There are {len(ori_files)} .ply files.')

        # div_files = sorted([f for f in os.listdir(args.output_dir) if '.ply' in f])
        # print(f'There are {len(div_files)} divided .ply files.')

        # os.makedirs(args.reconstructed_dir, exist_ok=True)
        # tqdm_handle = tqdm(ori_files)

        # for ori_file in tqdm_handle:
        #     merge_pc(ori_file, div_files, args.resolution, args.output_dir, os.path.join(args.reconstructed_dir, args.experiment))
            
    elif args.command == 'evaluate':
        assert os.path.exists(args.ori_dir), 'Input directory with original point cloud not found'
        assert os.path.exists(os.path.join(args.dec_dir, args.experiment)), 'Input directory with decompressed point clouds not found'
        assert os.path.exists(args.nor_dir), 'Input directory with original point cloud with normal attributes not found'
        assert os.path.exists(args.bin_dir), 'Input directory with compressed point cloud binaries not found'
        assert os.path.exists(args.pc_error_path), 'pc_error utility not found'
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        output_file = os.path.join(args.output_dir, args.experiment + '.csv')

        ori_files = sorted([f for f in os.listdir(args.ori_dir) if '.ply' in f])
        print(f'There are {len(ori_files)} .ply files.')

        nor_files = sorted([f for f in os.listdir(args.nor_dir) if '.ply' in f])
        print(f'There are {len(nor_files)} .ply files.')

        com_files = sorted([f for f in os.listdir(os.path.join(args.bin_dir, args.experiment)) if '.bin' in f])
        print(f'There are {len(com_files)} compressed .bin files.')

        res_dec_files = sorted([f for f in os.listdir(os.path.join(args.dec_dir, args.experiment)) if '.ply' in f])
        print(f'There are {len(res_dec_files)} restored decompressed .ply files.')

        tqdm_handle = tqdm(enumerate(ori_files), total=len(ori_files))

        bitrate_df = pd.DataFrame(data={
            'name': [],
            'points_in': [],
            'points_out': [],
            'bin_bytes': [],
            'bpp': [],
            'g_metric_D1': [],
            'g_metric_D2': [] })

        for idx, ori_file in tqdm_handle:
            ori_pc_path = os.path.join(args.ori_dir, ori_file)
            nor_pc_path = os.path.join(args.nor_dir, f'{ori_file[:-4]}_n.ply')
            res_dec_pc_path = os.path.join(os.path.join(args.dec_dir, args.experiment), res_dec_files[idx])
            
            ori_pc = PyntCloud.from_file(ori_pc_path)
            ori_num_points = ori_pc.points.shape[0]

            com_path = os.path.join(args.bin_dir, args.experiment, com_files[idx])
            com_bytes = os.stat(com_path).st_size
                
            res_dec_pc = PyntCloud.from_file(res_dec_pc_path)
            res_dec_points = res_dec_pc.points.shape[0]
            
            bpp = com_bytes * 8. / ori_num_points

            g_metric_D1, g_metric_D2 = evaluate_pc(ori_pc_path, res_dec_pc_path, nor_pc_path, args.resolution, args.pc_error_path)
            
            ori_num_points = int(ori_num_points)
            res_dec_points = int(res_dec_points)
            com_bytes = int(com_bytes)
            
            bitrate_df.loc[idx] = [ori_file, ori_num_points, res_dec_points, com_bytes, bpp, g_metric_D1, g_metric_D2]       

        pd.set_option('display.max_columns', None)    
        print(bitrate_df)
        bitrate_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)



