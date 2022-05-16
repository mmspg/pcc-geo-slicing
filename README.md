# Latent Space Slicing For Enhanced Entropy Modeling in Learning-Based Point Cloud Geometry Compression

- **Authors**: Nicolas Frank, Davi Lazzarotto and Touradj Ebrahimi.
- **Affiliation**: Multimedia Signal Processing Group (MMSPG), Ecole Polytechnique Fédérale de Lausanne.
- **Link**: https://ieeexplore.ieee.org/abstract/document/9747496

## Requirements

- Python 3.8.10
- Tensorflow 2.4.0
- Tensorflow-compression 2.0
- MPEG-PCC-dmetric-v0.13.5 for metric computation
- See requirements.txt


## Experiments

Every step of the point cloud compression, from training to evaluation, is done using the main file *point_cloud_compression_slice_conditioning.py*. The task to be executed is specified by passing the *task* keyword to the parser, which can be ***train, compress, decompress, evaluate***.

The global parser's arguments can be visualized using:

```bash
python point_cloud_compression_slice_conditioning.py -h
```


### Training

The model can be trained by passing the keyword *train* to the main script. A training dataset must be already partitioned into blocks with partition size given by `--resolution`. An example command can be seen down below, using lambda = 1000, 5 slices for latent partition and training block resolution of 64:

```bash
python point_cloud_compression_slice_conditioning.py --experiment 'PCC_5slices_lmbda1000' --model_path 'models/' train --train_glob 'train/blocks_64_rgb/*.ply' --resolution 64 --epochs 10 --lambda 1000 --num_slices 5
```

For a list of all the training parameters that can be changed, run:

```bash
python point_cloud_compression_slice_conditioning.py train -h
```


### Compression

A test set can be compressed either with a model trained by the user or with a pre-trained model that can be can be downloaded from the following **FTP** by using dedicated FTP clients, such as FileZilla or FireFTP (we recommend to use [FileZilla](https://filezilla-project.org/)):

```
Protocol: FTP
FTP address: tremplin.epfl.ch
Username: datasets@mmspgdata.epfl.ch
Password: ohsh9jah4T
FTP port: 21
```

The pre-trained models can be found in the folder **pcc-geo-slicing/models**.

In order to compress a test set, the user can pass the keyword *compress* to the main script. The command below is an example that can be used to compress a dataset with block resolution 128 using adaptive thresholding:

```bash
python point_cloud_compression_slice_conditioning.py --experiment 'PCC_5slices_lmbda1000' --model_path 'models/' compress --adaptive --resolution 128 --input_glob 'original/pointclouds_rgb/*.ply' --output_dir 'compressed/' 
```

Please note that a different block resolution can be used for training and for testing. 

For a list of all the compression parameters, run:

```bash
python point_cloud_compression_slice_conditioning.py compress -h
```

### Decompression

In order to compress a test set, the user can pass the keyword *decompress* to the main script. An example comand can be found here below:

```bash
python point_cloud_compression_slice_conditioning.py --experiment 'PCC_5slices_lmbda1000' --model_path 'models/' decompress --input_glob 'compressed/*.bin' --output_dir 'decompressed/'
```

For a list of all the decompression parameters, run:

```bash
python point_cloud_compression_slice_conditioning.py decompress -h
```


### Evaluation

The compression of point clouds can be evaluated through the computation of the bitrate per input point as well as through objective metrics D1 and D2 PSNR calculated between the original and decompressed point clouds. For that, the code from MPEG evaluation metric software must be built and the path to the generated binary should be specified by the parameter `--pc_error_path`. Moreover, versions of the original point clouds with normal vectors must available for the computation of the D2 PSNR metric. 

The following command can be used for the evaluation of the test dataset compressed with one compression model:

```bash
python point_cloud_compression_slice_conditioning.py --experiment 'PCC_5slices_lmbda1000' --model_path 'models/' evaluate --ori_dir 'original/pointclouds_rgb/' --dec_dir 'decompressed/' --nor_dir 'original/pointclouds_rgb_n/' --bin_dir 'compressed/' --output_dir 'results/' --pc_error_path 'pc_error' --resolution 1023
```

The resulting bitrate and metric values are stored in a csv file saved in the folder given by `--output_dir`. For a list of all the evaluation parameters, run:

```bash
python point_cloud_compression_slice_conditioning.py evaluate -h
```


## GPU usage for compression/decompression

It is recommended to use the CPU to compress/decompress rather than the GPU. Compressing/decompressing a block using a GPU can introduce floating-point round off error, that propagate and ruin the reconstruction. Compressing and decompressing using the same CPU alleviates this issue. More information can be found here:

J. Ballé, N. Johnston, D. Minnen,
["Integer Networks for data compression with latent-variable models"](https://openreview.net/pdf?id=S1zz2i0cY7)

If you wish to use the GPU anyway, use the keyword `--gpu` in the compress/decompress parser. Furthermore, the `--debug` argument in the decompress parser will try to avoid such reconstruction error by retrying the decompression until the distortion between the decompressed and the original block is under a threshold, which has to be set manually in the code.
