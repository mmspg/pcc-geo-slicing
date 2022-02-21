import scipy.spatial
import numpy as np
import tensorflow as tf
  
def pack_tensor(threshold, x_shape, y_shape, z_shape, z_string, *y_strings,  bytes_length = 2, adaptive = False):
    """
    Transforms the compression results into a bitstream for 
    transmission towards the decoder.
    
    Parameters:
       threshold: Threshold to transform the output of the NN into
                      an occupancy map.
         y_shape: Shape of the latent tensor.
         z_shape: shape of the tensor after passing through the
                  hyper-analysis transform
        z_string: String resulting from the compression of z.
       y_strings: Tuple of the slices string.
    bytes_length: Number of bytes to encode the data. Recommended 2.
        adaptive: Whether adaptive threshold is used.

    Output: A bitstream ready to be written to a file.
    """
    
    # Transform the shapes into strings.
    shape_strings = [bytes(x_shape.numpy().tolist()), bytes(y_shape.numpy().tolist()), bytes(z_shape.numpy().tolist())]
  
    # Convert the length of the slices into a string.
    if len(y_strings) != 1:
      length_slices = [len(y_string).to_bytes(bytes_length, byteorder = 'little') for y_string in np.squeeze(np.asarray(y_strings))]
      y_strings = np.squeeze(np.asarray(y_strings))
    else:
      length_slices = [len(np.squeeze(y_strings[0]).tolist()).to_bytes(bytes_length, byteorder = 'little')]
      y_strings = y_strings[0].numpy().tolist()
  
    #Concatenate every part together.
    bitstream = len(shape_strings[0]).to_bytes(1, byteorder = 'little') + \
    len(shape_strings[1]).to_bytes(1, byteorder = 'little') + \
    len(shape_strings[2]).to_bytes(1, byteorder = 'little') + \
    len(y_strings).to_bytes(1, byteorder = 'little') + b"".join(length_slices) + \
    len(z_string.numpy()[0]).to_bytes(bytes_length, byteorder = 'little') + \
    b"".join(shape_strings) +  z_string.numpy()[0] + b"".join(y_strings)
  
    # Add the threshold at the end if adaptive, else it is known to be 0.5.  
    assert threshold <= 1 and threshold >= 0, "Threshold outside specified range"
    if adaptive:
        threshold = (np.round(threshold.numpy() * 100)).astype('uint8')
        bitstream = bitstream + bytes([threshold])

    return bitstream
  
def unpack_tensor(bitstream, bytes_length = 2, adaptive = False):
    """
    Transforms the compression results into a bitstream for 
    transmission towards the decoder.
    
    Parameters:
    bitstream: Bitstream to decode.
 bytes_length: Number of bytes to encode the data. Recommended 2.
     adaptive: Whether adaptive threshold is used.

    Output: The decoded compression results.
    """
    
    # Recover #of characters of the shapes in bitstream.
    x_shape_length = bitstream[0]
    y_shape_length = bitstream[1]
    z_shape_length = bitstream[2]
  
    # Recover the slices metadata from the bitstream.
    num_slices = bitstream[3]
    slices_length = np.squeeze([np.fromstring(bitstream[4+(i-1)*bytes_length:i*bytes_length+4], dtype=np.int16).tolist() for i in range(1, num_slices+1)])
    if num_slices == 1:
      slices_length = np.expand_dims(slices_length, axis = -1)
  
    # Recover #of characters of the z string from the bitstream.
    length_z_string = int(np.fromstring(bitstream[num_slices*bytes_length+4:(num_slices+1)*bytes_length+4],dtype=np.int16))
  
    # Recover the shapes from the bitstream.
    x_shape = np.fromstring(bitstream[(num_slices+1)*bytes_length+4:][:x_shape_length], dtype=np.uint8)
    y_shape = np.fromstring(bitstream[(num_slices+1)*bytes_length+4+x_shape_length:][:y_shape_length], dtype=np.uint8)
    z_shape = np.fromstring(bitstream[(num_slices+1)*bytes_length+4+x_shape_length+y_shape_length:][:z_shape_length], dtype=np.uint8)
  
    # Recover the z string from the bitstream.
    z_string = [bitstream[(num_slices+1)*bytes_length+4+x_shape_length+y_shape_length+z_shape_length:][:length_z_string]]
  
    y_strings = []
    left = len(bitstream) - ((num_slices+1)*bytes_length+4+x_shape_length+y_shape_length+z_shape_length+length_z_string)

    # Recover the y strings from the bitstream.
    for i in range(num_slices):
      if i == 0:
        y_string = [bitstream[(num_slices+1)*bytes_length+4+x_shape_length+y_shape_length+z_shape_length+length_z_string:][:slices_length[i]]]
      else:
        y_string =[bitstream[(num_slices+1)*bytes_length+4+x_shape_length+y_shape_length+z_shape_length+length_z_string+np.sum(slices_length[:i]):][:slices_length[i]]]
      y_strings.append(y_string)
  
    # Recover the threshold from the bistream if necessary.
    if adaptive:
      threshold = bitstream[-1] / 100
      
      return (x_shape, y_shape, z_shape, z_string) + tuple(y_strings), threshold
    else:
      return (x_shape, y_shape, z_shape, z_string) + tuple(y_strings), 0.5

def po2po(block1_pc, block2_pc):
    """
    Compute the point to point (D1) metric between two blocks.
    
    Parameters:
    block1_pc: First block to compare.
    block2_pc: Second block to compare.
            
    Output: The D1 metric between block1_pc and block2_pc.
    """

    # A -> B point to point metric.
    block1_tree = scipy.spatial.cKDTree(block1_pc)
    nearest_ref1 = block1_tree.query(block2_pc)
    po2po_ref1_mse = (nearest_ref1[0] ** 2).mean()
  
    # B -> A point to point metric.
    block2_tree = scipy.spatial.cKDTree(block2_pc)
    nearest_ref2 = block2_tree.query(block1_pc)
    po2po_ref2_mse = (nearest_ref2[0] ** 2).mean()

    # D1 is the max between the two above.
    po2po_mse = np.max((po2po_ref1_mse, po2po_ref2_mse))

    return po2po_mse

def compute_optimal_threshold(model, tensors, pc, delta_t = 0.01, breakpt = 50, verbose = 1):
    """
    Computes the optimal threshold used to convert the output of the
    neural network into an occupancy map.
    
    Parameters:
    tensors: tuple containing the shapes and string
             that the model.compress function outputs.
    delta_t: Space between two consecutive threshold
             during the grid search.
    breakpt: Number of thresholds to try without improvement
    verbose: Level of verbosity. Either 0 (no printing),
             1 (partial printing) or 2 (full printing)
                 
    Output: The optimal threshold.
    """
    
    assert verbose in {0,1,2}, "Verbose should be either 0(no printing), 1 (partial printing) or 2 (full printing)"
    # Decompress the latent tensor.
    x_hat = tf.squeeze(model.decompress(*tensors))
    x_hat = x_hat.numpy()
            
    # Prepare parameters for search.
    num_not_improve = 0
    thresholds = tf.linspace(delta_t,1,tf.cast(tf.math.round(1/delta_t), dtype = tf.int64))
    min_mse = 1e10
    best_threshold = tf.constant(0)
            
    for threshold in thresholds:
                
        # Locate values above current test threshold.
        pa = np.argwhere(x_hat > threshold).astype('float32')
                
        # Compute the associated D1 metric.
        mse = po2po(pc, pa)
                
        # Empty PC cause D1 metric to be NaN.
        if np.isnan(mse):
            
            # Try having only one point in the middle of the
            # block, if this solution is better than the one
            # found so far, an empty block is the best solution.
            mean_pt = np.round(np.mean(pc, axis = 0))[np.newaxis, :]
            test_mse = po2po(pc, mean_pt)
            if verbose == 2:
                print(f'The D1 error for the mean point is {test_mse}, against {min_mse} for the current best threshold.')
                    
            # If the mean point is better than current threshold,
            # return an empty block (threshold = 1).
            if test_mse < min_mse:
                best_threshold = tf.constant(1)
                        
            # If the adaptive threshold finds a threshold too
            # low, return the fixed threshold.
            if best_threshold.numpy() < 0.1:
                best_threshold = tf.constant(0.5)
            if verbose >= 1:
                print(f' Best threshold found: {best_threshold.numpy()}')
            return best_threshold
                
        # Update the current best threshold if necessary.
        if mse < min_mse:
            min_mse = mse
            best_threshold = threshold
            num_not_improve = 0
            if verbose == 2:
                print(f'D1 mse value of {min_mse} found at t = {best_threshold.numpy()}')
        else:
            num_not_improve += 1
            if verbose == 2:
                print(f'Not a better threshold mse = {mse} at t = {threshold.numpy()}')
            if num_not_improve == breakpt:
                return best_threshold

