# Setting Up Data Paths

Expected dataset structure for ImageNet:

```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Expected dataset structure for CIFAR-10:

```
cifar10
|_ data_batch_1
|_ data_batch_2
|_ data_batch_3
|_ data_batch_4
|_ data_batch_5
|_ test_batch
|_ ...
```

Expected dataset structure for CIFAR-100:

```
cifar100
|_ file.txt~
|_ meta
|_ test
|_ train
```

Expected dataset structure for Tiny ImageNet:

```
tinyimagenet200
|_ test
|  |_ images
|     |_ test_0.JPEG
|     |_ ...
|_ train
|  |_ n01443537
|  |_ ...
|_ val
|  |_ images
|     |_ val_0.JPEG
|     |_ ...
|  |_ val_annotations.txt
|_ wnids.txt
|_ words.txt
```
