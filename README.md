# Latent Space Slicing For Enhanced Entropy Modeling in Learning-Based Point Cloud Geometry Compression

- **Authors**: Nicolas Frank, Davi Lazzarotto and Touradj Ebrahimi.
- **Affiliation**: Multimedia Signal Processing Group (MMSPG), Ecole Polytechnique Fédérale de Lausanne.
- **Link**: Coming soon.

## Requirements

- Python 3.8.10
- Tensorflow 2.4.0
- Tensorflow-compression 2.0
- MPEG-PCC-dmetric-v0.13.5 for metric computation
- See requirement.txt

## Getting Started

The algorithm uses point clouds partitioned in cubic blocks of various sizes. The convolutional nature of the model allows for the testing blocks to have a different size than the training ones.
The training and testing blocks are expected to be in separate folders, as well as the original unpartitioned test point clouds and their normals (optional, used to compute the point-to-plane D2 metrics). The MPEG-PCC-dmetric software must be placed in the same folder as the source code. The directory organisation should look similar to this:

- point_cloud_compression_slice_conditioning.py
- src
- pc_error
- train
  -  blocks_64_rgb
- test
  - blocks_128_rgb
- original
  - test_pointclouds_rgb
  - test_pointclouds_rgb_n

In the following, the directory organisation will refer to this one.


## Experiments

Every step of the point cloud compression, from training to evaluation, is done using the main file "point_cloud_compression_slice_conditioning.py". Specify which task you want to execute by passing the task keyword to the parser, included in ***train, compress, decompress, evaluate***.

The global parser's arguments can be accessed using:

`python point_cloud_compression_slice_conditioning.py -h`

and the ones of a specific task by inserting the task keyword before `-h`.


### Training

Assume you want to train a model on training blocks of size 64x64x64 with 5 slices, a rate-distortion parameter lambda of 1000 for 10 epochs. The trained model will be saved in a folder called 'models', under the name of 'PCC_5slices_lmbda1000'. The program should be started as:

`python point_cloud_compression_slice_conditioning.py '--experiment' 'PCC_5slices_lmbda1000' '--model_path' '/models/' 'train' '--train_glob' '/train/blocks_64_rgb/*.ply' '--resolution' '64' '--epochs' '10' '--lambda' '1000' '--num_slices' '5'`

For a list of all the training parameters that can be changed, run:

`python point_cloud_compression_slice_conditioning.py 'train' -h`


### Compression

After training your model, you can use it to compress point clouds. In this example we compress test blocks of size 128x128x128 that were not used during training, and compute the optimal threshold that will be used for decompression:

`python point_cloud_compression_slice_conditioning.py '--experiment' 'PCC_5slices_lmbda1000' '--model_path' '/models/' 'compress' '--adaptive' '--resolution' '128' '--input_glob' '/test/blocks_128_rgb/*.ply' '--output_dir' '/compressed/' `

For a list of all the compression parameters, run:

`python point_cloud_compression_slice_conditioning.py 'compress' -h`

### Decompression

After compressing, you can decompress the blocks and merge them back into a point cloud. In the previous step, the optimal threshold has been computed, so it is used in the decompression stage:

`python point_cloud_compression_slice_conditioning.py '--experiment' 'PCC_5slices_lmbda1000' '--model_path' '/models/' 'decompress' '--adaptive'  '--input_glob' '/compressed/*.tfci' '--output_dir' '/decompressed/' '--ori_dir' '/original/pointclouds_rgb/' '--reconstructed_dir' '/reconstructed/'`

For a list of all the decompression parameters, run:

`python point_cloud_compression_slice_conditioning.py 'decompress' -h`


### Evaluation

The reconstructed point clouds can now be evaluated. The resulting csv file is specified to be saved in a directory called 'Results', under the name of the experiment:

`python point_cloud_compression_slice_conditioning.py '--experiment' 'PCC_5slices_lmbda1000' '--model_path' '/models/' 'evaluate' '--ori_dir' '/original/pointclouds_rgb/' '--dec_dir' 'reconstructed/' '--nor_dir' '/original/pointclouds_rgb_n/' '--bin_dir' '/compressed/' '--output_dir' '/Results/'`

For a list of all the evaluation parameters, run:

`python point_cloud_compression_slice_conditioning.py 'evaluate' -h`

Afterwards, the directory organisation should look like this:

- point_cloud_compression_slice_conditioning.py
- src
- pc_error
- models
  - PCC_5slices_lmbda1000
    - ...
- Results
  - PCC_5slices_lmbda1000.csv
- train
  -  blocks_64_rgb
- test
  - blocks_128_rgb
- original
  - test_pointclouds_rgb
  - test_pointclouds_rgb_n
- compressed
  - ...
- decompressed
  - ...
- reconstructed
  - PCC_5slices_lmbda1000
    - ...


## GPU usage for compression/decompression

It is recommended to use the CPU to compress/decompress rather than the GPU. Compressing/decompressing a block using a GPU often introduces a floating-point round off error, that propagate and ruin the reconstruction. Compressing and decompressing using the same CPU alleviates this issue. More information can be found here:

J. Ballé, N. Johnston, D. Minnen,
["Integer Networks for data compression with latent-variable models"](https://openreview.net/pdf?id=S1zz2i0cY7)

If you wish to use the GPU anyway, use the keyword `'--gpu'` in the compress/decompress parser. Furthermore, the `'--debug'` argument in the decompress parser will try to get rid of the errors by retrying the decompression until they don't appear. The minimum tolerable error may have to be set manually, depending on the target bpp and the nature of the point cloud.
