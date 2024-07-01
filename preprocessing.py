from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest # para a Seleção Univariada
from sklearn.preprocessing import LabelEncoder

from abc import ABC, abstractmethod

class PreProcessor(ABC):
    '''Classe abstrata que contém operaçoes de um preprocessador de dados. '''
    @abstractmethod
    def preprocess(self,dataset):
        pass

class SelectKBestPreProcessor(PreProcessor):

    def __init__(self,k =4, seed =0, ratio=0.30):
        print("Inicializando seleção de atributos com k={}".format(k)) 
        self.seed = seed
        self.ratio = ratio
        self.k = k

    def __init_features(self, dataset):
        columns = dataset.columns.tolist()
        self.dataset = dataset
        self.features = columns[:-1]
        self.label = columns[-1]

    def __separate_feature_and_label(self):
        self.dataset.sort_index(inplace=True)
        x = self.dataset[self.features].values
        y = self.dataset[self.label].values
        return x,y
    
    def __fillna(self, col_name,value):
        self.dataset[col_name] = self.dataset[col_name].fillna(value)
    
    def __drop_unecessary_cols(self):
        self.df = self.df.drop(['Unnamed: 0', 'id'], axis=1)
        
    
    def __encode_categorical_columns(self):
        """
        Codifica colunas categóricas em um DataFrame substituindo os valores pelos números.

        :param df: DataFrame contendo os dados
        :return: DataFrame com colunas categóricas codificadas
        """
        label_encoders = {}
        df_encoded = self.dataset.copy()

        for column in df_encoded.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            label_encoders[column] = le
        self.labels = label_encoders
        self.dataset = df_encoded
        
    
    def __selectKBest(self,X,y):
        best_var = SelectKBest(score_func=f_classif, k=self.k)
        fit = best_var.fit(X, y)
        return fit.transform(X)
        
    def preprocess(self,dataset):
        self.__init_features(dataset)
        self.__fillna('Arrival Delay in Minutes',0.0)
        self.__encode_categorical_columns()
        X,y = self.__separate_feature_and_label()
        X = self.__selectKBest(X,y)
        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=self.ratio, random_state=self.seed)
        return x_train, x_test, y_train, y_test