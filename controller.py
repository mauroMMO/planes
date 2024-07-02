

from data_loader import DataLoader
from evaluator import Evaluator
from model import Model
from plotter import Plotter
from preprocessing import PreProcessor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class Controller():

    def __init__(self,dataLoader: DataLoader,preProcessor: PreProcessor,
                 models: Model, plotter: Plotter):
        self.__dataloader = dataLoader
        self.__preProcessor = preProcessor
        self.__models = models
        self.__plotter = plotter


    def __processModel(self,pipelines: list, model: Model, X_train, X_test, y_train, y_test):
        scoring = 'accuracy'
        num_particoes = 2
        for name, estimator in pipelines:
        
            print('\nGridSearching {}'.format(name))
            kfold = StratifiedKFold(n_splits=num_particoes, shuffle=True, 
                                    random_state=self.__preProcessor.seed)
            grid_search = GridSearchCV(estimator=estimator, param_grid=model.grid, scoring=scoring, cv=kfold)
            grid_search.fit(X_train, y_train)
            print('\n{} - Melhor: {} usando {}'.format(name, grid_search.best_score_, grid_search.best_params_))
            
            estimator.set_params(**grid_search.best_params_)
            
            #Model Training
            print('\nTraining for {}'.format(name))
            estimator.fit(X_train, y_train)
            
            #Model Predict
            predictions = estimator.predict(X_test)
            #Evaluation
            print('\nEstimation for {}'.format(name))
            model.evaluate(y_test,predictions)
            self.__plotter.plot(y_test,predictions)

    
    def run(self):
        #Load Dataset
        dataset = self.__dataloader.dataset()
        #Preprocessing
        x_train, x_test, y_train, y_test = self.__preProcessor.preprocess(dataset)

        for model in self.__models:
            model_tuple = (model.name,model.model)
            pipelines = [('{}'.format(model.name), Pipeline(steps=[model_tuple]) )]
            self.__processModel(pipelines, model, x_train, x_test, y_train, y_test)
        
        