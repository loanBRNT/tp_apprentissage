import numpy as np
from termcolor import colored

date = 20220928


def CalculerIndividusCentresReduits(Individus, CentresGravite):
    # ---------------------------------------------------------------------------------
    # function [IndividusCentresReduits]=CalculerIndividusCentresReduits(Individus, CentresGravite);
    #
    # Calcul des individus centrés réduits
    #
    # ---------------------------------------------------------------------------------
    # Entree : Individus                 [NbrIndividus x NbrParametres]
    #          CentresGravite            [NbrClasses   x NbrParametres]
    #
    # Sortie : IndividusCentresReduits   [NbrIndividus x NbrParametres]
    # ---------------------------------------------------------------------------------
    NbrIndividus, NbrVariables = np.shape(Individus)

    IndividusCentres = Individus - np.ones((NbrIndividus, 1)) * CentresGravite[0, :].reshape((1, NbrVariables))
    CT = np.dot(IndividusCentres.T, IndividusCentres) / NbrIndividus
    IndividusCentresReduits = np.multiply(IndividusCentres, np.dot(np.ones((NbrIndividus, 1)),
                                                                   np.power(np.diag(CT), -0.5).reshape(
                                                                       (1, NbrVariables))))

    return IndividusCentresReduits


def CalculerCentresGravite(Individus, NoClasses):
    # ---------------------------------------------------------------------------------
    # function [CentresGravite]=CalculerCentresGravite(Individus, NoClasses);
    #
    # Calcul des centres de gravité de chaque classe.
    #
    # ---------------------------------------------------------------------------------
    # Entree : Individus                [NbrIndividus x NbrParametres]
    #          NoClasses                [NbrIndividus]
    #
    # Sortie : CentresGravite           [NbrClasses   x NbrParametres]
    #          Nb : CentreGravite[0] =  Centre gravite global
    # ---------------------------------------------------------------------------------

    NbrIndividus, NbrVariables = np.shape(Individus)
    NbrClasses = np.max(NoClasses) + 1

    CentresGravite = np.zeros((NbrClasses, NbrVariables))
    CentresGravite[0] = np.mean(Individus, axis=0)
    for q in range(1, NbrClasses):
        IndClasse = np.argwhere(NoClasses == q)[:, 0]
        CentresGravite[q] = np.mean(Individus[IndClasse], axis=0)
    return CentresGravite


def CalculerVariances(Individus, NoClasses, CentresGravite):
    # ---------------------------------------------------------------------------------
    # function [VT, VA, VE]=CalculerVariances(Individus, NoClasses, CentresGravite);
    #
    # Calcul des variances Totale, Intraclasses et Interclasses
    # ---------------------------------------------------------------------------------
    # Entree : Individus          [NbrIndividus x NbrParametres]
    #          NoClasses          [NbrIndividus x 1 ]
    #          CentresGravite     [NbrClasses   x NbrParametres]
    #
    # Sortie : VT = Variance totale       [1 x 1]
    #          VA = Variance intraclasses [1 x 1]
    #          VE = Variance interclasses [1 x 1]
    # ----------------------------------------------------------------------------------

    NbrIndividus, NbrVariables = np.shape(Individus)
    NbrClasses = np.max(NoClasses) + 1

    VA = 0
    VE = 0
    if NbrClasses != 1:
        ### Variance intraclasses
        VA = 0
        for q in range(1, NbrClasses):
            IndClasse = np.argwhere(NoClasses == q)[:, 0]
            LngIndClasse = np.size(IndClasse)
            vect = Individus[IndClasse] - np.ones((LngIndClasse, 1)) * CentresGravite[q]
            VA += np.trace(np.dot(vect.T, vect))
        VA /= NbrIndividus

        ### Variance interclasses
        VE = 0
        for q in range(1, NbrClasses):
            LngIndClasse = np.size(np.argwhere(NoClasses == q)[:, 0])
            vect = CentresGravite[q] - CentresGravite[0]
            VE += LngIndClasse * np.dot(vect.T, vect)
        VE /= NbrIndividus

    ### Variance totale

    vect = Individus - np.ones((NbrIndividus, 1)) * CentresGravite[0, :]
    VT = np.trace(np.dot(vect.T, vect)) / NbrIndividus

    ### Vérification

    if (abs(VT - VA - VE) > 1e-9) and (NbrClasses != 1):
        print(colored(
            'Attention pb de calcul pour les variances (VT<>VA+VE)!\n VT={0}, VA={1}, VE={2}, difference={3}'.format(VT,
                                                                                                                     VA,
                                                                                                                     VE,
                                                                                                                     VT - VA - VE),
            'red'))

    return VT, VA, VE


def CalculerMatricesCovariance(Individus, NoClasses, CentresGravite):
    # ---------------------------------------------------------------------------------
    #  function [CT, CA, CE]=CalculerMatricesCovariance(Individus, NoClasses, CentresGravite);
    #
    # Calcul des matrices de covariance Totale, Intraclasses et Interclasses
    # ---------------------------------------------------------------------------------
    # Entree : Individus          [NbrIndividus x NbrParametres]
    #          NoClasses          [NbrIndividus x 1 ]
    #          CentresGravite     [NbrClasses   x NbrParametres]
    #
    # Sortie : CT = Matrice de covariance totale       [NbrParametres x NbrParametres]
    #          CA = Matrice de covariance intraclasses [NbrParametres x NbrParametres]
    #          CE = Matrice de covariance interclasses [NbrParametres x NbrParametres]
    # ----------------------------------------------------------------------------------

    NbrIndividus, NbrVariables = np.shape(Individus)
    NbrClasses = np.max(NoClasses) + 1

    # Variables centrées

    IndividusCentres = Individus - np.dot(np.ones((NbrIndividus, 1)), CentresGravite[0, :].reshape((1, NbrVariables)))

    if NbrClasses == 1:

        CA = np.zeros((NbrVariables, NbrVariables))
        CE = np.zeros((NbrVariables, NbrVariables))
    else:
        # Variables centrées par classe

        TabMoyenneClasse = np.zeros((NbrIndividus, NbrVariables))
        for q in range(1, NbrClasses):
            IndClasse = np.argwhere(NoClasses == q)[:, 0]
            LngIndClasse = np.size(IndClasse)
            TabMoyenneClasse[IndClasse] = np.dot(np.ones((LngIndClasse, 1)),
                                                 CentresGravite[q, :].reshape((1, NbrVariables)))

        IndividusCentresClasse = Individus - TabMoyenneClasse;

        # Matrice de covariance intraclasses

        CA = np.dot(IndividusCentresClasse.T, IndividusCentresClasse) / NbrIndividus

        # Matrice de covariance interclasses

        CE = np.zeros((NbrVariables, NbrVariables))
        for q in range(1, NbrClasses):
            IndClasse = np.argwhere(NoClasses == q)[:, 0]
            LngIndClasse = np.size(IndClasse)
            vect = (CentresGravite[q] - CentresGravite[0]).reshape((NbrVariables, 1))
            CE += LngIndClasse * np.dot(vect, vect.T)
        CE /= NbrIndividus

    # Matrice de covariance totale

    CT = np.dot(IndividusCentres.T, IndividusCentres) / NbrIndividus

    # Vérifications

    if (abs(np.sum(np.sum(CA + CE - CT))) > 1e-9) and (NbrClasses != 1):
        t, a, e = CalculerVariances(Individus, NoClasses, CalculerCentresGravite(Individus, NoClasses));

        if abs(np.trace(CT) - t) > 1e-9:
            print(colored('Attention pb de calcul sur la matrice de covariance totale CT!\n', 'red'))
            CT = np.zeros((NbrVariables, NbrVariables))
        if abs(np.trace(CA) - a) > 1e-9:
            print(colored('Attention pb de calcul sur la matrice de covariance intraclasses CA!\n', 'red'))
            CA = np.zeros((NbrVariables, NbrVariables))
        if abs(np.trace(CE) - e) > 1e-9:
            print(colored('Attention pb de calcul sur la matrice de covariance interclasses CE!\n', 'red'))
            CE = np.zeros((NbrVariables, NbrVariables))

    return CT, CA, CE
