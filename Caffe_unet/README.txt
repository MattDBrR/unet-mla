# Caffe U-Net 2015 : Segmentation Cellulaire

Ce projet implémente et teste le modèle **U-Net (2015)** pour la segmentation cellulaire en utilisant le dataset **ISBI DIC-HeLa**. Le modèle a été exécuté à l'aide de l'implémentation **Caffe U-Net** d'origine, et les résultats ont été comparés à des masques de vérité terrain.

## Objectifs du projet

1. Tester l'**implémentation originale de l’U-Net (2015)** sur des images microscopiques.
2. **Évaluer les performances** du modèle pré-entraîné sur le dataset **ISBI DIC-HeLa**.
3. **Mesurer les scores Dice** et **IoU** pour évaluer la qualité de la segmentation générée par le modèle.

## Prérequis

### Environnement

- **Python 3.x**
- **Caffe** (CPU support)
- **tifffile**, **opencv-python**, **scikit-image**, **numpy**, **torch**
installer les dépendances avec :

```bash
pip install -r requirements.txt
```

conda env create -f environment.yml
conda activate unet2015

éléchargement des données

Le dataset ISBI DIC-HeLa utilisé dans ce projet peut être téléchargé depuis le site officiel du challenge ISBI : ISBI DIC-HeLa Dataset
.

Les images d’entrée doivent être placées dans data/images/, et les masques de vérité terrain dans data/masks/.

Modèle

Le modèle utilisé est l'implémentation originale de l’U-Net (2015) en Caffe. Il a été pré-entraîné sur un dataset générique de cellules, mais les poids pour ISBI DIC-HeLa ne sont pas disponibles, donc nous avons utilisé un modèle générique déjà pré-entraîné.

Téléchargement des poids pré-entraînés

Les poids pré-entraînés pour Caffe U-Net sont disponibles ici : Caffe U-Net Pretrained Weights
.

Décompresse les fichiers dans le répertoire caffemodels/ de ce projet.

Tester le modèle

Pour tester le modèle U-Net 2015 sur une image donnée et générer les prédictions, exécute le script run_unet_caffe.py :
python run_unet_caffe.py \
  --deploy /path/to/deploy.prototxt \
  --weights /path/to/weights.caffemodel \
  --input /path/to/input_image.tif \
  --output /path/to/output_mask.png

