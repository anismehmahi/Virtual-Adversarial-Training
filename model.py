import torch.nn as nn

# Définition de la classe VAT (Virtual Adversarial Training)
class VAT(nn.Module):

    # Initialisation du modèle
    def __init__(self, top_bn=True):
        super(VAT, self).__init__()
        # Indique si une normalisation BatchNorm sera appliquée en sortie
        self.top_bn = top_bn

        self.main = nn.Sequential(
            # Première couche de convolution : 3 canaux d'entrée, 128 canaux de sortie
            nn.Conv2d(3, 128, 3, 1, 1, bias=False),  # Filtre 3x3, stride 1, padding 1
            nn.BatchNorm2d(128),  # Normalisation des lots (BatchNorm) pour stabiliser l'apprentissage
            nn.LeakyReLU(0.1),  # Fonction d'activation LeakyReLU avec un léger passage pour les valeurs négatives

            # Deuxième couche de convolution : identique à la première mais sur 128 canaux
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            # Troisième couche de convolution : identique mais toujours sur 128 canaux
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            # MaxPooling : réduction de la taille spatiale avec une fenêtre 2x2
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),  # Applique un dropout spatial pour réduire le surapprentissage

            # Quatrième couche de convolution : 128 canaux d'entrée, 256 canaux de sortie
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # Cinquième couche de convolution
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # Sixième couche de convolution
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # MaxPooling et Dropout spatial
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),

            # Septième couche de convolution : 256 canaux d'entrée, 512 canaux de sortie
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),  # Pas de padding
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            # Couche de réduction dimensionnelle avec un filtre 1x1
            nn.Conv2d(512, 256, 1, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # Autre couche de réduction dimensionnelle avec un filtre 1x1
            nn.Conv2d(256, 128, 1, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            # Pooling adaptatif pour réduire chaque carte de caractéristiques à une taille 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Couche linéaire pour produire les 10 classes de sortie
        self.linear = nn.Linear(128, 10)
        # BatchNorm appliqué à la sortie finale si top_bn est activé
        self.bn = nn.BatchNorm1d(10)

    # Fonction de passage avant (forward)
    def forward(self, input):
        # Passage de l'entrée à travers les couches principales
        output = self.main(input)
        # Mise à plat des caractéristiques spatiales et passage à travers la couche linéaire
        output = self.linear(output.view(input.size()[0], -1))
        # Application de BatchNorm à la sortie finale si top_bn est activé
        if self.top_bn:
            output = self.bn(output)
        return output
