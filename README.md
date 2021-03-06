# InvCLR: Improving Transformation Invariance in Contrastive Representation Learning
This is a [PyTorch](https://github.com/pytorch/pytorch) implementation of the ICLR paper [Improving Transformation Invariance in Contrastive Representation Learning (InvCLR)](https://arxiv.org/abs/2010.09515):
```
@article{foster2020improving,
  title={Improving Transformation Invariance in Contrastive Representation Learning},
  author={Foster, Adam and Pukdee, Rattana and Rainforth, Tom},
  journal={arXiv preprint arXiv:2010.09515},
  year={2020}
}
```

## Installation and datasets
Install PyTorch following the instructions [here](https://pytorch.org/). Download the the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.
The Spirograph dataset is included in this code base.
To use the Spirograph dataset on its own, see this [standalone repo](https://github.com/rattaoup/spirograph).
To install the requisite packages for this project, use `pip install -r requirements.txt`.
Note: to install `torchlars` it is necessary to set the environment variable `CUDA_HOME`.

Make a note about `dataset-paths.json`

## Training an encoder
We support multi-GPU `DataParallel` training.
Use the following command to train an encoder from scratch on CIFAR-10. We ran this using 8 GPUs.
```bash
$ python3 invclr.py \
  --num-epochs 1000 \
  --cosine-anneal \
  --arch resnet50 \
  --dataset cifar10 \
  --lambda-gp 1e-1 \
  --filename cifar10_run
```
Set `--dataset cifar100` to train on CIFAR-100.
To train an encoder on the Spirograph dataset, use
```bash
$ python3 invclr.py \
  --num-epochs 50 \
  --cosine-anneal \
  --test-freq 0 \
  --save-freq 10 \
  --arch resnet18 \
  --dataset spirograph \
  --lambda-gp 1e-2 \
  --filename spirograph_run \
  --gp-upper-limit 1000
```
You can set `--lambda-gp 0` to train an encoder with no gradient penalty.

## Evaluating an encoder
Use the following command to evaluate the trained CIFAR-10 encoder on untransformed inputs with 50% of
the training labels used for supervized training
```bash
$ python3 eval.py \
  --load-from cifar10_run_epoch999.pth \
  --untransformed \
  --proportion 0.5
```
for Spirograph, we used
```bash
$ python3 eval.py \
  --load-from spirograph_run_epoch049.pth \
  --reg-weight 1e-8 \
  --proportion 0.5
```
On Spirograph, you can view regression results for each of the separate tasks by adding `--componentwise`, otherwise
the presented loss is the mean over all tasks.

### Feature averaging
Use the following command to evaluate classification performance of feature averaging using an average of 100 samples
```bash
$ python3 eval.py \
  --load-from cifar10_run_epoch999.pth \
  --num-passes 100 
```
for  Spirograph, run the following code
```bash
$ python3 eval.py \
  --load-from spirograph_run_epoch049.pth \
  --num-passes 100 \
  --reg-weight 1e-8
```
We obtained the following
| Dataset    | Loss | Accuracy |
|------------|------|----------|
| CIFAR-10   |      |          |
| CIFAR-100  |      |          |
| Spirograph |      |          |

