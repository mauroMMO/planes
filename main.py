from preprocessing import SelectKBestPreProcessor
from data_loader import DataLoaderFromLocal

# Instancia o carregador de dados
data_loader = DataLoaderFromLocal('planes_1.csv', 'planes_2.csv')


dataset = data_loader.dataset()

# Verifica as colunas do dataset
columns = dataset.columns.tolist()

# Instancia o preprocessador
preprocessor = SelectKBestPreProcessor()

# Preprocessa o dataset
x_train, x_test, y_train, y_test = preprocessor.preprocess(dataset)

# Exibe informações sobre os conjuntos de dados
print("Tamanho do conjunto de treino:", x_train.shape)
print("Tamanho do conjunto de teste:", x_test.shape)
print("Primeiras linhas do conjunto de treino X:")
print(x_train[:5])
print("Primeiras linhas do conjunto de treino Y:")
print(y_train[:5])