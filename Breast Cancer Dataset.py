#!/usr/bin/env python
# coding: utf-8

# # Prueba Tecnica Celsia

# ### Importando librerías

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import mglearn
import matplotlib.pyplot as plt


# ### Se decide dentro de las opciones de no supervisado y el no poder utilizar k-means (k-medoids, k-modes ...) usar PCA (analisis de componentes principales) ; Se eliminan columnas innecesarias para el modelo que son el contador de filas y se considera que el diagnostico ya que al querer aplicar un modelo no supervisado se está dando por sentado que la informacion no tiene categorias ya que es lo que un modelo no supervisado pretende realizar un agrupamiento por caracteristicas nuevas
# 

# ###  se estandarizan los valores del dataset, para poder ingresarlos al modelo

# In[2]:


cancer = pd.read_csv("Analitica/wdbc1.csv",sep=",")
cancer_features = cancer.drop(['ID','diagnosis'], axis=1) 
scaler = StandardScaler()
scaler.fit(cancer_features)
X_scaled = scaler.transform(cancer_features)


# ###  1 significa Benignos y 0 significa malignos segun el documento por la cantidad descrita allí, la cual se verifica a continuacion

# In[3]:


cancer.diagnosis.value_counts() 


# ### Se crea label para la gráfica segun diagnositco

# In[4]:


diagnosis_names = np.array(['maligno', 'benigno'])


# ###  Verificando el numero de componentes requeridas (hiperparametro) para describir los datos usando la razon de la varianza acumulativa x # componentes

# In[5]:


pca = PCA().fit(cancer_features)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('número de componentes')
plt.ylabel('varianza acumulativa')
plt.show();


# ### Según la grafica observamos que las 2 primeras componentes contienen alrededor del 99% de la varianza, por lo que se considera mas que suficiente para el modelo

# ###  Se aplica el modelo con 2 componentes

# In[6]:


pca = PCA(n_components=2)

pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print("Forma original: {}".format(str(X_scaled.shape)))
print("Forma reducida: {}".format(str(X_pca.shape)))


# ### Se grafican los componentes en un espacio bidimensional

# In[7]:


plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.diagnosis.values)
plt.legend(diagnosis_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show();


# ### Se verifica con un regresor logistico que el comportamiento de separacion en los datos es aproximadamente lineal

# In[8]:


clf = LogisticRegression(solver="liblinear")
clf.fit(X_pca,cancer.diagnosis.values)

mglearn.plots.plot_2d_classification(clf,X_pca,fill=True, cm=ListedColormap(['blue','orange']))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.diagnosis.values)
plt.show();


# ### Resultados de la clasificacion con PCA 2 componentes, se utiliza la regla de 80/20 para definir la cantidad de datos de prueba, y se usa  un random state para obtener resultados reproducibles para que se ejecute el programa

# In[9]:


X_train, X_test, y_train, y_test=train_test_split(X_pca,cancer.diagnosis.values,
                                                  test_size=0.2,random_state=10)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print (classification_report(y_test,y_pred))
print (confusion_matrix(y_test,y_pred))


# ### Se puede entonces concluir que para este conjunto de datos, haber tomado solo dos componentes principales permite visualizar que la separación de los datos es lineal sin perder calidad en los resultados. Y que el modelo obtuvo buenos resultados expresados en las metricas anteriores
