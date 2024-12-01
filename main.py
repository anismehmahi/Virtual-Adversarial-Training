import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os
from sklearn.metrics import recall_score, f1_score

# Configuration des hyperparamètres
batch_size = 32  # Taille des lots pour l'entraînement
eval_batch_size = 100  # Taille des lots pour l'évaluation
unlabeled_batch_size = 128  # Taille des lots pour les données non étiquetées
num_labeled = 100  # Nombre d'exemples étiquetés
num_valid = 1000  # Nombre d'exemples de validation
num_iter_per_epoch = 400  # Nombre d'itérations par époque
eval_freq = 5  # Fréquence d'évaluation (tous les 5 époques)
lr = 0.001  # Taux d'apprentissage
cuda_device = "0"  # Identifiant du périphérique CUDA

# Définition des arguments en ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Choisir le dataset (mnist)')
parser.add_argument('--dataroot', required=True, help='Chemin vers les données')
parser.add_argument('--use_cuda', type=bool, default=True, help='Utilisation de CUDA')
parser.add_argument('--num_epochs', type=int, default=120, help='Nombre d\'époques')
parser.add_argument('--epoch_decay_start', type=int, default=80, help='Début de la décroissance du taux d\'apprentissage')
parser.add_argument('--epsilon', type=float, default=2.5, help='Valeur de l\'epsilon pour la régularisation VAT')
parser.add_argument('--top_bn', type=bool, default=True, help='Application de BatchNorm sur la sortie')
parser.add_argument('--method', default='vat', help='Méthode d\'entraînement (VAT)')

opt = parser.parse_args()

# Configuration du périphérique CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

# Fonction pour transférer les données vers CUDA si activé
def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x

# Fonction d'entraînement du modèle
def train(model, x, y, ul_x, optimizer):
    ce = nn.CrossEntropyLoss()  # Définition de la fonction de perte pour les données étiquetées
    y_pred = model(x)  # Prédictions sur les données étiquetées
    ce_loss = ce(y_pred, y)  # Calcul de la perte d'entropie croisée

    ul_y = model(ul_x)  # Prédictions sur les données non étiquetées
    v_loss = vat_loss(model, ul_x, ul_y, eps=opt.epsilon)  # Calcul de la perte VAT
    loss = v_loss + ce_loss  # Perte totale

    # Ajout de la perte d'entropie si spécifié
    if opt.method == 'vatent':
        loss += entropy_loss(ul_y)

    # Calcul des gradients
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step() 

    return v_loss, ce_loss

# Fonction d'évaluation du modèle
def eval(model, x, y):
    y_pred = model(x) 
    prob, idx = torch.max(y_pred, dim=1)  # Classe avec la probabilité maximale
    return torch.eq(idx, y).float().mean()  # Calcul de la précision moyenne

# Initialisation des poids pour le modèle
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # Initialisation pour les couches convolutionnelles
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # Initialisation pour BatchNorm
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:  # Initialisation pour les couches linéaires
        m.bias.data.fill_(0)

# Chargement des datasets
if opt.dataset == 'mnist':
    num_labeled = 100  # Nombre d'exemples étiquetés pour MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))  # Normalisation spécifique à MNIST
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=opt.dataroot, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))  # Normalisation spécifique à MNIST
                      ])),
        batch_size=eval_batch_size, shuffle=True)

else:
    raise NotImplementedError  # Gestion des autres datasets non implémentée

train_data = []
train_target = []

for (data, target) in train_loader:
    if opt.dataset == 'mnist':
        data = data.expand(-1, 3, -1, -1)  # Expansion à 3 canaux pour correspondre à l'architecture
    train_data.append(data)
    train_target.append(target)

train_data = torch.cat(train_data, dim=0)
train_target = torch.cat(train_target, dim=0)

# Division des données en validation et entraînement
valid_data, train_data = train_data[:num_valid, ], train_data[num_valid:, ]
valid_target, train_target = train_target[:num_valid], train_target[num_valid:, ]

# Division en données étiquetées et non étiquetées
labeled_train, labeled_target = train_data[:num_labeled, ], train_target[:num_labeled, ]
unlabeled_train = train_data[num_labeled:, ]

# Initialisation du modèle, des poids et de l'optimiseur
model = tocuda(VAT(opt.top_bn))
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Boucle d'entraînement
for epoch in range(opt.num_epochs):
    if epoch > opt.epoch_decay_start:
        # Décroissance du taux d'apprentissage
        decayed_lr = (opt.num_epochs - epoch) * lr / (opt.num_epochs - opt.epoch_decay_start)
        optimizer.lr = decayed_lr
        optimizer.betas = (0.5, 0.999)

    for i in range(num_iter_per_epoch):
        # Sélection de lots aléatoires pour l'entraînement
        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        batch_indices_unlabeled = torch.LongTensor(np.random.choice(unlabeled_train.size()[0], unlabeled_batch_size, replace=False))
        ul_x = unlabeled_train[batch_indices_unlabeled]

        v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                optimizer)

        # Affichage des pertes toutes les 100 itérations
        if i % 100 == 0:
            print("Epoch:", epoch, "Iter:", i, "VAT Loss:", v_loss.item(), "CE Loss:", ce_loss.item())

    # Évaluation périodique
    if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:
        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
        print("Train accuracy:", train_accuracy.item())

        for (data, target) in test_loader:
            if opt.dataset == 'mnist':
                data = data.expand(-1, 3, -1, -1)
            test_accuracy = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
            print("Test accuracy:", test_accuracy.item())
            break


# Évaluation finale avec calcul des métriques
test_accuracy = 0.0  # Initialisation de la précision globale du test
counter = 0  # Compteur pour le nombre total d'exemples

# Initialisation des compteurs par classe
true_positive = [0] * 10  # Vrai positif (TP) pour chaque classe
false_positive = [0] * 10  # Faux positif (FP) pour chaque classe
false_negative = [0] * 10  # Faux négatif (FN) pour chaque classe

# Parcours de toutes les données de test
for (data, target) in test_loader:
    if opt.dataset == 'mnist':
        data = data.expand(-1, 3, -1, -1)  # Expansion des données à 3 canaux pour correspondre au modèle
    n = data.size()[0]  # Nombre d'exemples dans le lot actuel
    
    # Passage des données dans le modèle pour obtenir les prédictions
    outputs = model.eval()
    preds = torch.argmax(outputs(Variable(tocuda(data))), dim=1)  # Indices des prédictions les plus probables

    # Calcul de la précision pour ce lot
    acc = eval(outputs, Variable(tocuda(data)), Variable(tocuda(target)))
    test_accuracy += n * acc  # Mise à jour de la précision cumulée
    counter += n  # Mise à jour du compteur d'exemples

    # Mise à jour des compteurs TP, FP, FN par classe
    for t, p in zip(target.numpy(), preds.cpu().numpy()):
        if t == p:  # Si la prédiction est correcte
            true_positive[t] += 1
        else:  # Sinon, mise à jour des faux positifs et négatifs
            false_positive[p] += 1
            false_negative[t] += 1

# Calcul du rappel (recall) pour chaque classe
recall_per_class = [
    tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Rappel = TP / (TP + FN)
    for tp, fn in zip(true_positive, false_negative)
]

# Calcul du F1-Score pour chaque classe
f1_per_class = [
    (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0  # F1 = 2 * TP / (2 * TP + FP + FN)
    for tp, fp, fn in zip(true_positive, false_positive, false_negative)
]

# Calcul des métriques pondérées globales
total_true = sum(true_positive)  # Nombre total de vrais positifs
weighted_recall = sum(tp * r for tp, r in zip(true_positive, recall_per_class)) / total_true  # Rappel pondéré
weighted_f1 = sum(tp * f for tp, f in zip(true_positive, f1_per_class)) / total_true  # F1-score pondéré

# Affichage des résultats finaux
print("Full test accuracy:", test_accuracy.item() / counter)  # Précision totale sur l'ensemble de test
print("Recall per class:", recall_per_class)  # Rappel pour chaque classe
print("Weighted Recall:", weighted_recall)  # Rappel global pondéré
print("F1 Score per class:", f1_per_class)  # F1-Score pour chaque classe
print("Weighted F1 Score:", weighted_f1)  # F1-Score global pondéré
