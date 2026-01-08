# U-Net PyTorch – Entraînement, inférence et évaluation

Ce dossier contient **ma contribution au projet** :  
l’entraînement d’un modèle **U-Net en PyTorch** pour la segmentation d’images,
ainsi que l’évaluation quantitative de ses performances sur l’ensemble du
dataset à l’aide des métriques **Dice** et **IoU**.

L’objectif est de fournir un **modèle de référence (baseline)** permettant
la comparaison avec d’autres approches de segmentation, notamment le
modèle U-Net Caffe (2015) pré-entraîné.

---

## Travail réalisé

Les étapes suivantes ont été implémentées et exécutées :

1. Entraînement d’un modèle U-Net en PyTorch sur le dataset du projet  
2. Génération des masques de segmentation pour toutes les images du dataset  
3. Évaluation des prédictions sur **l’ensemble du dataset** à l’aide des
   métriques Dice et Intersection over Union (IoU)

---
## Dataset

Le dataset **est inclus** dans ce dépôt GitHub.

Les masques peuvent contenir plusieurs valeurs entières ; ils sont convertis en masques binaires (premier plan / arrière-plan) lors de l’évaluation.

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader).

---

## Pré-entraînement et transfer learning

Le modèle U-Net utilise un **encodeur initialisé avec des poids pré-entraînés
sur ImageNet**. Ce pré-entraînement permet de disposer de représentations
visuelles génériques (bords, textures, motifs) et améliore la convergence
lors de l’entraînement.

L’ensemble du réseau (encodeur + décodeur) est ensuite **fine-tuné sur le
dataset de segmentation du projet**. Les performances finales reflètent donc
l’apprentissage sur les données spécifiques au problème étudié, et non une
segmentation directe à partir d’ImageNet.

A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5) 
```
---

## Entraînement

L’entraînement du modèle est effectué via le script `train.py`.

`console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

---
### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pth`.
---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)

