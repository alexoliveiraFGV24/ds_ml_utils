import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def Fisher_li_discriminant_binary(points:np.array, true_labels:np.array) -> np.array:

    # Definindo as classes
    C1 = points[true_labels == 0]
    C2 = points[true_labels == 1]
    
    N1 = len(C1)
    N2 = len(C2)
    N = len(points)

    # Calculando a média de cada classe
    m1 = np.sum(C1, axis=0) / N1
    m2 = np.sum(C2, axis=0) / N2
    m1 = m1.reshape(-1,1)
    m2 = m2.reshape(-1,1)

    # Calculando a matriz de covariância total (dentro de cada classe)
    Sw = sum((x.reshape(-1,1)-m1) @ (x.reshape(-1,1)-m1).T for x in C1) + sum((y.reshape(-1,1)-m2) @ (y.reshape(-1,1)-m2).T for y in C2)

    # Calculando w
    w = np.linalg.inv(Sw) @ (m2 - m1)  # Seria proporcional, mas não importa (só queremos a direção)
    w = w.reshape(-1,1)  # Para manter a forma de matriz coluna

    # Calculando o valor de w_0
    m = (N1 * m1 + N2 * m2) / N
    w_0 = -w.T @ m
    w_0 = w_0[0,0]  # Para ter certeza que é um escalar

    return w, w_0

def accuracy_ridge(points:np.array, true_labels:np.array, desc:str, w:np.array, w_0:float) -> float:
    
    r = RidgeClassifier()
    projection = points @ w + w_0  # Projeção dos pontos em w (com bias w_0)
    projection = projection.reshape(-1, 1)
    r.fit(projection, true_labels)
    predicted_projection = r.predict(projection)
    accuracy = accuracy_score(predicted_projection, true_labels)  # Acurácia da projeção (acertos/todos)
    
    return f"projection: {desc}, accuracy = {accuracy}"

def accuracy_lda(X_train, y_train, X_test, y_test) -> float:
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=1)
    lda.fit(X_train, y_train)
    accuracy_lda = lda.score(X_test, y_test)
    
    return f"accuracy lda = {accuracy_lda}"

def plot_projection_binary(points:np.array, true_labels:np.array, project_vector:np.array, w_0:float, project_vector_name:str, x_label:str, hue:str='class', stat:str='count'):
    
    # Criando o DataFrame com os pontos e as classes e a projeção
    df = pd.DataFrame({'x': points[:,0], 'y':points[:,1], 'class':true_labels})
    df[project_vector_name] = df[['x', 'y']].values @ project_vector + w_0

    # Plotando os pontos na projeção da direção de w
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.histplot(df, x=project_vector_name, hue=hue, bins=30, ax=ax, stat=stat)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.show()
    
def plot_projection(X:np.array, y:np.array):
    # Criando o modelo LDA e projetando os dados na direção ótima
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=1)
    lda.fit(X, y)
    X_projection = lda.transform(X)
    y_pred = lda.predict(X)
    X_projection = X_projection.flatten()  # Transformando em vetor
    y_pred = y_pred.flatten()  # Transformando em vetor

    # Montando o DataFrame
    # Não usei minha função por que a saída é diferente dá que eu tinha montado
    df = pd.DataFrame({'projection': X_projection, 'class':y_pred})

    # Plotando os dados
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.histplot(df, x='projection', hue='class', bins=30, alpha=0.5, ax=ax, stat='count')
    plt.xlabel("Projection on w")
    plt.ylabel("Count")
    plt.show()

def plot_line(df:pd.DataFrame, points:np.array, true_labels:np.array):

    # Aprendendo a superfície de decisão e o valor de w_0
    w, w_0 = Fisher_li_discriminant_binary(points, true_labels)
    
    # Gerando os pontos para desenhar a linha
    t = np.linspace(-1000, 1000, 100).reshape(-1,1)
    x = w[0] * t + w_0
    y = w[1] * t + w_0
    
    # Plotando a linha e os pontos
    _, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(df, x='x', y='y', hue='class', alpha=0.5, ax=ax)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.plot(x, y, "--", color='red', label="solution")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.xlabel("")
    plt.ylabel("")
    leg = ax.legend()
    for handle in leg.legend_handles:
        handle.set_alpha(1)
    plt.show()
    