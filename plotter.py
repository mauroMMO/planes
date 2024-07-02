import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Plotter(ABC):
    """ Classe abstrata que contém operações do Plotter. """

    @abstractmethod
    def plot(self,y_test,predictions):
        """ Método que plota o gráfico. """
        pass

class MatPlotter(Plotter):
    def __init__(self):
        self.__xlabel = "Samples"
        self.__ylabel = "y and predictions"
        self.__title = "y and predictions comparison"
        self.__legend = "upper left"

    def plot(self,y_test,predictions):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(predictions,"-b", label="y_hat")
        plt.plot(y_test,"-r", label="y")
        x1,x2,y1,y2 = plt.axis()
        x1 = 0
        x2 = 50
        plt.axis((x1,x2,y1,y2))
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.title(self.__title)
        plt.legend(loc=self.__legend)
        plt.show()