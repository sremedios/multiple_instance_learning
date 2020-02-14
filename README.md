# Multiple Instance Learning
Deep learning for binary classification using multiple instance learning (MIL).

Notes:
- Binary classifier: positive vs negative.
- For MIL, bags contain instances.  Bags have a ground truth, either positive or negative.
Instances within the bag will be classified as positive or negative.  A bag is only negative
if all its instances are negative.  If one or more instances are positive, then the bag is
considered positive.
- MIL Max Pooling means we only calculate loss/gradients for the most-positive instance. 

## Directions:
### Directory Setup:
1. Create data directories and subdirectories as below.

```
+-- {DATA_DIR}/
|   +-- positive/
|   |   +-- my_positive_file_1.nii.gz
|   |   +-- my_positive_file_2.nii.gz
|   |   +-- [...]
|   |   +-- my_positive_file_n.nii.gz
|   +-- negative/
|   |   +-- my_negative_file_1.nii.gz
|   |   +-- my_negative_file_2.nii.gz
|   |   +-- [...]
|   |   +-- my_negative_file_m.nii.gz
```

2. Run `python make_tfrecord.py {DATA_DIR} TARGETDIR` and supply the correct arguments.
3. Run `python train.py` 


## References
If this code is useful for your project, please cite our work:

https://arxiv.org/abs/1911.05650

```
Remedios, Samuel W., Zihao Wu, Camilo Bermudez, Cailey I. Kerley, Snehashis Roy, Mayur B. Patel, John A. Butman, Bennett A. Landman, and Dzung L. Pham. "Extracting 2D weak labels from volume labels using multiple instance learning in CT hemorrhage detection." arXiv preprint arXiv:1911.05650 (2019).
```
