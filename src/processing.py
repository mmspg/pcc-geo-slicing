import numpy as np
import tensorflow as tf
from src import pc_io
"""
Data loading and processing.
Code adpated from https://github.com/mauriceqch/pcc_geo_cnn
"""

def load_pc(args):
    """
    Loads the PC's blocks from the files.
                
    Parameters:
    args: Arguments from the parser.
                                                  
    Output: The blocks point information, separated in training
    and validation, and the blocks shape information.
    """
        
    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args.resolution, args.channels_last)
    files = pc_io.get_files(args.train_glob)
    perm = np.random.permutation(len(files))
    points = pc_io.load_points(files[perm][:args.num_data], p_min, p_max)
    points_train = points[:-args.num_val]
    points_val = points[-args.num_val:]
    return points_train, points_val, dense_tensor_shape

def pc_to_tf(points, dense_tensor_shape, channels_last=False):
    """
    Transforms the loaded points of the blocks into 
    a tensorflow sparse tensor.
        
    Parameters:
                points: Points loaded using the 'load_pc' function.
    dense_tensor_shape: Shape of the dense tensor.
         channels_last: True if the channel dimension is the last one,
                        else False.
                                          
    Output: A sparse tensorflow tensor containing the 
            block's geometry information.
    """
                        
    x = points
        
    paddings = [[0, 0], [1, 0]] if not channels_last else [[0, 0], [0, 1]]
    geo_indices = tf.pad(x[:, :3], paddings, constant_values=0)
        
    indices = tf.cast(geo_indices, tf.int64)
    values = tf.ones_like(x[:, 0])
                
    st = tf.sparse.SparseTensor(indices, values, dense_tensor_shape)

    return st

def process_x(x, dense_tensor_shape):
    """
    Creates the dense tensor to be used with the
    neural network from its sparse form.
        
    Parameters:
                     x: Sparse tensor to be made dense.
    dense_tensor_shape: Shape of the dense tensor.
                                          
    Output: The dense version of the sparse tensor.
    """
        
    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)

    return x
        
def create_dataset(features, dense_tensor_shape, args, repeat=True):
    """
    Creates a tf.data.Dataset from raw point cloud's blocks.
        
    Parameters:
              features: Raw loaded blocks.
    dense_tensor_shape: Shape of the dense block.
                  args: Arguments from the parser.
                repeat: Whether to make the dataset infinite or not.
                        Recommended for the training set only.
                                          
        Output: a tf.data.Dataset ready for training.
    """
        
    # Create input data pipeline.
    with tf.device('/cpu:0'):
        zero = tf.constant(0)
        
        # Create the dataset from the raw data.
        dataset = tf.data.Dataset.from_generator(lambda: iter(
                    features), tf.float32, tf.TensorShape([None, 3]))
                        
        # Make the dataset infinite if repeat is True.
        if repeat:
            dataset = dataset.shuffle(buffer_size=len(features))
            dataset = dataset.repeat()
        
        # Transform the raw data into a dense tf tensor to be used
        # with the neural network.
        dataset = dataset.map(lambda x: pc_to_tf(x, dense_tensor_shape,
                                args.channels_last), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x: process_x(x, dense_tensor_shape), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Creates batches of data.
        dataset = dataset.batch(args.batch_size, drop_remainder=False)
    return dataset


