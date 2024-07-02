from controller import Controller
from preprocessing import SelectKBestPreProcessor
from data_loader import DataLoaderFromLocal
from plotter import MatPlotter
import sys
from model import LogisticRegressionClassifier, KNNClassifier, NaiveBayesClassifier, SVMClassifier, BaggingClassifierModel, RandomForestClassifierModel, ExtraTreesClassifierModel, GradientBoostingClassifierModel




with open('nb.txt', 'w') as f:
    sys.stdout = f
    # Instancia o carregador de dados
    data_loader = DataLoaderFromLocal('planes_1.csv', 'planes_2.csv')



    preprocessor = SelectKBestPreProcessor()

    matPlotter = MatPlotter()

    models = [
        
        #KNNClassifier(),
        NaiveBayesClassifier(),
        #BaggingClassifierModel(),
        #RandomForestClassifierModel(),
        #ExtraTreesClassifierModel(),
        #GradientBoostingClassifierModel()
    ]

    controller = Controller(data_loader,preprocessor,models,matPlotter)
    controller.run()

    sys.stdout = sys.__stdout__
