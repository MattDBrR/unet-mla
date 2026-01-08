U-Net PyTorch:

-Modèle U-Net entraîné sur le dataset du projet

-Encodeur initialisé avec des poids pré-entraînés sur ImageNet (transfer learning)

-Fine-tuning complet du réseau sur les données de segmentation

-Génération des masques de segmentation pour l’ensemble du dataset

-Évaluation quantitative sur tout le dataset à l’aide des métriques Dice et IoU

U-Net Caffe (2015):

-Utilisation du modèle U-Net original pré-entraîné (Ronneberger et al., 2015)

-Inférence uniquement (aucun ré-entraînement) sur le dataset du projet

-Génération des masques de segmentation pour toutes les images

-Évaluation sur le même dataset et avec les mêmes métriques (Dice, IoU)