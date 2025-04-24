import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay

# Carregar o dataset
df = pd.read_csv('heart_cleveland_upload.csv')

# Separar os dados em variáveis independentes (X) e dependente (y)
X = df.drop("condition", axis=1)
y = df["condition"]

# Dividir em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Função para treinar e avaliar o modelo
def fit_and_score(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    return {"KNN": score}

# Executar a função
result = fit_and_score(X_train, X_test, y_train, y_test)
print("Resultado do fit_and_score:", result)

# Listas para armazenar os scores
train_scores = []
test_scores = []

# Valores para n_neighbors
neighbors = range(1, 21)

# Instanciar o KNN
knn = KNeighborsClassifier()

# Loop para testar diferentes valores de n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(neighbors, train_scores, label="Score de Treino", marker='o')
plt.plot(neighbors, test_scores, label="Score de Teste", marker='s')
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Valor ajustado para n_neighbors")
plt.ylabel("Desempenho do modelo")
plt.title("Desempenho do KNN para diferentes valores de k")
plt.legend()
plt.grid(True)
plt.show()

# Exibir o melhor desempenho
print(f"Desempenho máximo de KNN nos dados de teste: {max(test_scores)*100:.2f}%")
