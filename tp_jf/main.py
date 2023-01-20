import os
import scipy
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from FichiersEtudiants.FichiersEtudiants.TPClassif import *
import features


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
        vecteurCoef = vecteurCoef[0, 1:11]
        mat.append(vecteurCoef)

    mat = np.array(mat).reshape((nbFic, -1))
    listNumClasse = np.array(listNumClasse).T
    return mat, listNumClasse


def afd(X_train, y_train, X_test, y_test):
    # Calcul des vnouvelles valeurs pour train
    centreGravite = CalculerCentresGravite(X_train, y_train)
    X_train_centreReduit = CalculerIndividusCentresReduits(X_train, centreGravite)
    centreGraviteReduit_train = CalculerCentresGravite(X_train_centreReduit, y_train)

    # Calcul des nouvelles valeurs pour test
    centreGravite = CalculerCentresGravite(X_test, y_test)
    X_test_centreReduit = CalculerIndividusCentresReduits(X_test, centreGravite)

    CT, CA, CE = CalculerMatricesCovariance(X_train_centreReduit, y_train, centreGraviteReduit_train)
    CACE = np.dot(np.linalg.inv(CA), CE)
    l, u = np.linalg.eig(CACE)
    # choix des lambda
    #M = np.array(u[:, 0:10])

    # nouvelle donne
    return np.dot(X_train_centreReduit, M), np.dot(X_test_centreReduit, M)


def acp(X_train, y_train, X_test, y_test):
    # Calcul des vnouvelles valeurs pour train
    centreGravite = CalculerCentresGravite(X_train, y_train)
    X_train_centreReduit = CalculerIndividusCentresReduits(X_train, centreGravite)
    centreGraviteReduit_train = CalculerCentresGravite(X_train_centreReduit, y_train)

    # Calcul des nouvelles valeurs pour test
    centreGravite = CalculerCentresGravite(X_test, y_test)
    X_test_centreReduit = CalculerIndividusCentresReduits(X_test, centreGravite)

    # Matrice de passage
    CT, CA, CE = CalculerMatricesCovariance(X_train_centreReduit, y_train, centreGraviteReduit_train)
    L, u = np.linalg.eig(CT)

    #M = np.array(u[:, 2:6])

    return np.dot(X_train_centreReduit, M), np.dot(X_test_centreReduit, M)


def evaluationPerf(dataTrain, etiquetteTrain, dataTest, etiquetteTest, typeModele="MLPClassifier"):
    if typeModele == "KMeans":
        classif = KMeans(init="k-means++", n_clusters=10, n_init=40, random_state=0)
    elif typeModele == "MLPClassifier":
        classif = MLPClassifier(random_state=0, hidden_layer_sizes=(128, 128), max_iter=500, early_stopping=True)
    elif typeModele == "SVC":
        classif = SVC()
    elif typeModele == "RandomForestClassifier":
        classif = RandomForestClassifier(max_depth=5)
    else:
        print("Modele inconnu")
        return None
    classif.fit(dataTrain, etiquetteTrain)
    pred = classif.predict(dataTest)
    score = accuracy_score(etiquetteTest, pred)
    return confusion_matrix(etiquetteTest, pred), score


# lecture
data_train, listNum_train = analyseSignaux("./Signaux/train/",
                                           ['aa', 'ee', 'eh', 'eu', 'ii', 'oe', 'oh', 'oo', 'uu', 'yy'])
data_test, listNum_test = analyseSignaux("./Signaux/test/",
                                         ['aa', 'ee', 'eh', 'eu', 'ii', 'oe', 'oh', 'oo', 'uu', 'yy'])

# afd
data_train_afd, data_test_afd = afd(data_train, listNum_train, data_test, listNum_test)

# acp
data_train_acp, data_test_acp = acp(data_train, listNum_train, data_test, listNum_test)

# eval
print(evaluationPerf(data_train, listNum_train, data_test, listNum_test, typeModele="RandomForestClassifier"))
print(evaluationPerf(data_train_afd, listNum_train, data_test_afd, listNum_test, typeModele="RandomForestClassifier"))
print(evaluationPerf(data_train_acp, listNum_train, data_test_acp, listNum_test, typeModele="RandomForestClassifier"))
