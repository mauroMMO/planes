import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
class Plotter(ABC):
    """ Classe abstrata que contém operações do Plotter. """

    @abstractmethod
    def plot(self,y_test,predictions,label_encoder):
        """ Método que plota o gráfico. """
        pass
import matplotlib.pyplot as plt


class MatPlotter(Plotter):
    def __init__(self):
        self.__xlabel = "Amostras"
        self.__ylabel = "valores reais e previsões"
        self.__title = "Comparação entre valores reais e previsões"
        self.__legend = "upper left"
                
    def plot(self, y_test, predictions, sample_size=100):
        
        if len(y_test) > sample_size:
            indices = np.random.choice(len(y_test), sample_size, replace=False)
            y_test = y_test[indices]
            predictions = predictions[indices]

        plt.figure(figsize=(10, 6)) 
        plt.plot(predictions, "-b", label="Previsões") 
        plt.plot(y_test, "-r", label="Valores Reais") 
        
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.title(self.__title)
        plt.legend(loc=self.__legend)
        plt.tight_layout() 
        plt.show()