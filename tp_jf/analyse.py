import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import FichiersEtudiants.FichiersEtudiants.TPClassif as TPClassif
import features

CheminFichiersTrain = "./Signaux/train/"
CheminFichiersTest = "./Signaux/test/"
Prefixe = ['aa', 'ee', 'eh', 'eu', 'ii', 'oe', 'oh', 'oo', 'uu', 'yy']


def extraction(chemin, prefixe):
    data = []
    listeNumClasse = []
    nbfichiers = 0
    for NomFichier in os.listdir(chemin):
        nbfichiers += 1
        (Fe, Echantillons) = scipy.io.wavfile.read(chemin + NomFichier)
        listeNumClasse.append(1 + prefixe.index(NomFichier[0:2]))
        VecteurCoefficients = features.mfcc(Echantillons, Fe, winstep=len(Echantillons) / Fe,
                                            winlen=len(Echantillons) / Fe)
        data.append(VecteurCoefficients)

    data = np.array(data).reshape((nbfichiers, -1))
    listeNumClasse = np.array(listeNumClasse).T

    return data, listeNumClasse


def AFD(data, NoClasses):
    centreGravites = TPClassif.CalculerCentresGravite(data, NoClasses)
    CT, CA, CE = TPClassif.CalculerMatricesCovariance(data, NoClasses, centreGravites)
    valeurp, vectp = np.linalg.eig(np.dot(np.linalg.inv(CA), CE))
    afd = np.dot(data, vectp[:, :6])
    plt.plot(afd, '+')
    plt.show()


def Classification(datatrain, etiquetteTrain, datatest, etiquetteTest, typeclass):
    if typeclass == "Kmeans":
        classif = KMeans(n_clusters=10)
    elif typeclass == "MLPC":
        classif = MLPClassifier()

    elif typeclass == "SVC":
        classif = SVC()

    elif typeclass == "RFC":
        classif = RandomForestClassifier()
    else:
        classif = None

    classif.fit(datatrain, etiquetteTrain)
    y_predict = classif.predict(datatest)
    return sklearn.metrics.confusion_matrix(etiquetteTest, y_predict)


datatrain, etiquettedata = extraction(CheminFichiersTrain, Prefixe)
datatest, etiquettetest = extraction(CheminFichiersTest, Prefixe)

print(Classification(datatrain, etiquettedata, datatest, etiquettetest, 'RFC'))
