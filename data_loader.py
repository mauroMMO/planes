from abc import ABC, abstractmethod
import os

import pandas as pd

class DataLoader(ABC):
    '''Classe abstrata que contém operações de um carregador de dados.'''

    @abstractmethod
    def dataset(self):
        '''Método que retorna um dataset.'''
        pass
    
class DataLoaderFromLocal(DataLoader):
    '''Inicializador concatena os dois arquivos passados.'''
    def __init__(self,file_name_1,file_name_2):
        planes_1 = pd.read_csv(file_name_1)
        planes_2 = pd.read_csv(file_name_2)
        self.__planes =  pd.concat([planes_1, planes_2])

    def dataset(self):
        return self.__planes