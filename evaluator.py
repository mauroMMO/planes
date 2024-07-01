from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from abc import ABC, abstractmethod

class Evaluator(ABC):
    """Classe abstrata que contém operações de um Avaliador"""

    @abstractmethod
    def evaluate(self,y_test,predictions):
        """ Método que printa o resutltado das metricas de avaliação """
        pass

class ClassifierEvaluator(Evaluator):
    """ Classe que contém as operações necessarias para avaliar o modelo de Classificação """

    def calculate_accuracy_score(self,y_test,predictions):
        """ Método que retorna o accuracy score """
        return accuracy_score(y_test, predictions)
    
    def calculate_precision_score(self,y_test,predictions):
        """ Método que retorna o precision score """
        return precision_score(y_test,predictions)
    
    def calculate_recall_score(self,y_test,predictions):
        """ Método que retorna o recall score """
        return recall_score(y_test,predictions)
    
    def calculate_f1_score(self,y_test,predictions):
        """ Método que retorna o f1 score """
        return f1_score(y_test,predictions)
    
    def evaluate(self,y_test,predictions):
        print('accuracy:',self.calculate_accuracy_score(y_test,predictions))
        print('precision:',self.calculate_precision_score(y_test,predictions))
        print('recall:',self.calculate_recall_score(y_test,predictions))
        print('f1-score:',self.calculate_f1_score(y_test,predictions))