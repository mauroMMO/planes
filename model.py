from sklearn.linear_model import LinearRegression

from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from evaluator import ClassifierEvaluator

from enum import Enum

class ModelType(Enum):
    """Enumerator that represents the model types and their respective evaluators."""

    CLASSIFICATION = ClassifierEvaluator()


class Model(ABC):
    """Classe abstrata que contém operacoes de um Modelo."""

    def fit(self, x_train, y_train):
        """Método para realizacao de treinamento."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """Método para realizacao de predicoes."""
        return self.model.predict(x_test)
    

    @abstractmethod
    def type(self):
        """Method that returns type of a Model, ModelType."""
        pass

    def evaluate(self, y_test, predictions):
        """Method that performs the evaluation of the model."""
        self.type().value.evaluate(y_test, predictions)


class LogisticRegressionClassifier(Model):
    def __init__(self, max_iter=200):
        print("Setting Up LogisticRegressionClassifier")
        self.name = "LR"
        self.model = LogisticRegression(max_iter=max_iter)
    grid = {
        'LR__max_iter': [100, 200, 300, 1000, 2000, 3000],
        'LR__C': [0.1, 1.0, 10.0]
    }
    def type(self):
        return ModelType.CLASSIFICATION 
    
class KNNClassifier(Model):
    def __init__(self):
        print("KNNClassifier")
        self.name = "KNN"
        self.model = KNeighborsClassifier()
    grid = {
        'KNN__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
        'KNN__weights': ['uniform', 'distance'],
        'KNN__metric': ["euclidean", "manhattan", "minkowski"],
    }
    def type(self):
        return ModelType.CLASSIFICATION 

class NaiveBayesClassifier(Model):
    def __init__(self):
        print("NaiveBayesClassifier")
        self.name = "NB"
        self.model = GaussianNB()
    grid = {}
    def type(self):
        return ModelType.CLASSIFICATION 


class SVMClassifier(Model):
    def __init__(self):
        print("SVMClassifier")
        self.name = "SVM"
        self.model = SVC()
    grid = {
        'SVM__C': [0.1, 1.0],
        'SVM__kernel': ['linear', 'rbf']
    }
    def type(self):
        return ModelType.CLASSIFICATION 


class BaggingClassifierModel(Model):
    def __init__(self):
        print("BaggingClassifierModel")
        self.name = "Bag"
        self.model = BaggingClassifier()
    grid = {
        'Bag__n_estimators': [50, 100, 200],
        'Bag__max_samples': [0.5, 0.75, 1.0]
    }
    def type(self):
        return ModelType.CLASSIFICATION 


class RandomForestClassifierModel(Model):
    def __init__(self):
        print("RandomForestClassifierModel")
        self.name = "RF"
        self.model = RandomForestClassifier()
    grid = {
        'RF__n_estimators': [50, 100, 200],
        'RF__max_depth': [None, 5, 10]
    }
    def type(self):
        return ModelType.CLASSIFICATION 


class ExtraTreesClassifierModel(Model):
    def __init__(self):
        print("ExtraTreesClassifierModel")
        self.name = "ET"
        self.model = ExtraTreesClassifier()
    grid = {
        'ET__n_estimators': [50, 100, 200],
        'ET__max_depth': [None, 5, 10]
    }
    def type(self):
        return ModelType.CLASSIFICATION 


class GradientBoostingClassifierModel(Model):
    def __init__(self):
        print("GradientBoostingClassifierModel")
        self.name = "GB"
        self.model = GradientBoostingClassifier()
    grid = {
        'GB__n_estimators': [50, 100, 200],
        'GB__learning_rate': [0.1, 0.5, 1.0]
    }
    def type(self):
        return ModelType.CLASSIFICATION 


