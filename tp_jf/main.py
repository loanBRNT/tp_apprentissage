import os
import scipy
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from FichiersEtudiants.FichiersEtudiants.TPClassif import *
from tp_jf import features


def analyseSignaux(cheminFic, prefixe):
    mat = []
    listNumClasse = []
    nbFic = 0
    for fic in os.listdir(cheminFic):
        nbFic += 1
        # Recup des infos du fichier
        Fe, Echantillon = scipy.io.wavfile.read(cheminFic + fic)
        listNumClasse.append(prefixe.index(fic[0:2]) + 1)

        '''
        #affichage
        plt.figure(1)
        plt.plot(Echantillon)
        plt.title('Representation temporelle du signal')
        dsp = np.abs(np.fft.fft(Echantillon))
        plt.figure(2)
        plt.plot(dsp)
        plt.title('Densite spectrale de Puissance')
        plt.show()
        '''

        # evaluation des perfs
        vecteurCoef = features.mfcc(Echantillon, Fe, winlen=len(Echantillon) / Fe, winstep=len(Echantillon) / Fe)
        vecteurCoef = vecteurCoef[0,1:11]
        mat.append(vecteurCoef)

    mat = np.array(mat).reshape((nbFic, -1))
    listNumClasse = np.array(listNumClasse).T.reshape(len(listNumClasse), 1)
    return mat, listNumClasse


def afd(X, y):
    # Calcul des valeurs et vecteurs propres des donnees
    centreReduit = CalculerCentresGravite(X, y)
    CT, CA, CE = CalculerMatricesCovariance(X, y, centreReduit)
    CACE = np.dot(np.linalg.inv(CA), CE)
    l, u = np.linalg.eig(CACE)

    # choix des lambda
    u = np.array(u[:, 0:6])

    # nouvelle donne
    return np.dot(X, u)


def evaluationPerf(dataTrain, etiquetteTrain, dataTest, etiquetteTest, typeModele="KMeans"):
    if typeModele == "KMeans":
        classif = KMeans(init="k-means++", n_clusters=10, n_init=40, random_state=0)
    elif typeModele == "MLPClassifier":
        classif = MLPClassifier(random_state=0, hidden_layer_sizes=(128, 128), max_iter=500, early_stopping=True)
    elif typeModele == "SVC":
        classif = SVC(gamma=2, C=1)
    elif typeModele == "RandomForestClassifier":
        classif = RandomForestClassifier(max_depth=5)
    else:
        print("Modele inconnu")
        return None
    classif.fit(dataTrain, etiquetteTrain)
    pred = classif.predict(dataTest)
    print(classif.score(dataTest,etiquetteTest))
    return confusion_matrix(etiquetteTest, pred)


# lecture
data_train, listNum_train = analyseSignaux("./Signaux/train/",
                                           ['aa', 'ee', 'eh', 'eu', 'ii', 'oe', 'oh', 'oo', 'uu', 'yy'])
data_test, listNum_test = analyseSignaux("./Signaux/test/",
                                         ['aa', 'ee', 'eh', 'eu', 'ii', 'oe', 'oh', 'oo', 'uu', 'yy'])


# afd
data_train_reformat = afd(data_train, listNum_train)
data_test_reformat = afd(data_test, listNum_test)


plt.figure(3)
plt.plot(data_train_reformat, "+")
plt.show()

# eval
print(evaluationPerf(data_train_reformat, listNum_train, data_test_reformat, listNum_test, typeModele="MLPClassifier"))
