import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Fonction pour calculer la divergence KL entre deux log-probabilités
def kl_div_with_logit(q_logit, p_logit):
    # Applique la fonction softmax aux logits de q
    q = F.softmax(q_logit, dim=1)
    # Applique la fonction log-softmax aux logits de q
    logq = F.log_softmax(q_logit, dim=1)
    # Applique la fonction log-softmax aux logits de p
    logp = F.log_softmax(p_logit, dim=1)

    # Calcul du terme q*logq (entropie de q)
    qlogq = (q * logq).sum(dim=1).mean(dim=0)
    # Calcul du terme q*logp (cross-entropie entre q et p)
    qlogp = (q * logp).sum(dim=1).mean(dim=0)

    # Retourne la divergence KL : H(q) - H(q, p)
    return qlogq - qlogp


# Fonction pour normaliser un vecteur selon la norme L2
def _l2_normalize(d):
    d = d.numpy()
    # Divise chaque vecteur par sa norme L2
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    # Retourne le tenseur normalisé
    return torch.from_numpy(d)


# Fonction pour calculer la perte VAT (Virtual Adversarial Training)
def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):
    # Initialisation d'un vecteur perturbateur aléatoire d
    d = torch.Tensor(ul_x.size()).normal_()

    # Effectue plusieurs itérations pour raffiner la perturbation
    for i in range(num_iters):
        # Normalise d et multiplie par xi (petite valeur)
        d = xi * _l2_normalize(d)
        # Convertit d en une variable nécessitant un calcul de gradient
        d = Variable(d.cuda(), requires_grad=True)
        # Calcule la sortie du modèle avec la donnée perturbée
        y_hat = model(ul_x + d)
        # Calcule la divergence KL entre les probabilités prédites et celles de la donnée originale
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        # Effectue une rétropropagation pour obtenir le gradient de la perte
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        # Réinitialise les gradients du modèle
        model.zero_grad()

    # Normalise la perturbation finale d
    d = _l2_normalize(d)
    d = Variable(d.cuda())
    # Calcule la perturbation virtuelle adv
    r_adv = eps * d

    # Calcule la perte basée sur la divergence KL avec la donnée perturbée
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


# Fonction pour calculer la perte d'entropie
def entropy_loss(ul_y):
    # Calcule les probabilités en appliquant softmax
    p = F.softmax(ul_y, dim=1)
    # Calcule l'entropie : - somme des p * log(p)
    return -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)
