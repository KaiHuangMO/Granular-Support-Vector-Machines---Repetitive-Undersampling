
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

class gsvm(object):
    def __init__(self, C = 100,
                 T = 10,class_weight = 'balanced',
                 degree = 3,gamma='auto', kernel='rbf'
                 ):
        self.C = C
        self.T = T
        self.class_weight = class_weight
        self.degree = degree
        self.gamma = gamma
        self.kernel = kernel
        self.allSVC = SVC(C = self.C, class_weight=self.class_weight,
                      degree=self.degree, gamma = self.gamma,
                      kernel= self.kernel)

    def rebuild(self, xTrain, yTrain, sv, xNLSV):  # rebuild SVC
        xNew = []
        yNew = []
        count = 0
        for i in range(0, len(yTrain)):
            if yTrain[i] == 1:
                xNew.append(xTrain[i])
                yNew.append(yTrain[i])
            else:
                if i not in sv:
                    xNew.append(xTrain[i])
                    yNew.append(yTrain[i])
                    count += 1
                else:
                    xNLSV.append(xTrain[i])
        return xNew, yNew, xNLSV, count

    def fit(self, x, y):
        #
        xPos = []
        xNeg = []
        xTrain = []
        yTrain = []
        xlastTrain = []
        ylastTrain = []

        for i in range(0, len(y)):
            if y[i] == 1:
                xPos.append(x[i])
                xlastTrain.append(x[i])
                ylastTrain.append(y[i])
            else:
                xNeg.append(x[i])
            xTrain.append(x[i])
            yTrain.append(y[i])
        xNLSV = []
        iterRecord = 0
        for i in range(0, self.T):
            svc = SVC(C = self.C, class_weight=self.class_weight,
                      degree=self.degree, gamma = self.gamma,
                      kernel= self.kernel)
            print (iterRecord)
            iterRecord += 1
            svc.fit(xTrain, yTrain)
            sv = svc.support_  # This is support vector
            xTrain, yTrain, xNLSV, lastMar = self.rebuild(xTrain, yTrain, sv, xNLSV)  # rebuild sample
            #print (lastMar)
            if lastMar < 0.1 * len(xPos):
                break

        for i in xNLSV:
            xlastTrain.append(i)
            ylastTrain.append(0)

        self.allSVC.fit(xlastTrain, ylastTrain)

    def predict(self, x):
        return  self.allSVC.predict(x)


