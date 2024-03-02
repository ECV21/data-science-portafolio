# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:41:55 2023

@author: eduar
"""

# Data preprocessing, analysis and visualization

# 1) Data preprocessing
# 2) Analyzing data
# 3) Visualizing data-univariate plots
# 4) Visualizing data-multivariate plots



# ML algorithms don´t work so well with processing raw data, in other words we
# must apply some transformations on it. With data preprocessing, we convert
# raw data into a clean data set.
#   1) Recaling data
#   2) Mean removal
#   3) Standardizing data
#   4) One hot encoding
#   5) Normalizing data
#   6) Labe encoding
#   7) Binarizing data


# Rescaling data
# For data with attributes of varying scales, we can rescale attributes to possess
# the same scale. We rescale attributes into the range 0 to 1 and call it
# normalization. We use the MinMaxScaler class from scikit-lern.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#carda de la base
base = pd.read_csv("C:/Users/eduar/OneDrive/Escritorio/Ejercicios Python/datos_ejercicios.csv")
base.head()

#selecciona solo columnas a usar
base1 = base[["ANIO_GASTO", "MES_GASTO", "Edad", "Gasto",
              "Tipo_empleado"]]

#convertir columna de tipo_empleado en binario:
base1["Tipo_empleado"].unique   # Activo / Jubilado
base1["Tipo_empleado"] = base1["Tipo_empleado"].apply(lambda x: 1 if x=="Activo" else 0)
base1["Tipo_empleado"].unique         

#crear otro dataframe pero sin valores null
base2 = base1.dropna()
base2.head()

#obtener solo valores del dataframe, es decir, sin los encabezados
array = base2.values #valores de un DataFrame como un objeto NumPy array.
array

#separating data into input an output components
x = array[:,0:4 ] #crear x, con valores de las primeras 3 columnas de la matriz
x
y = array[:, 4] #crear y, con los valores de la columna 4
y

#aplicar técnica
scaler = MinMaxScaler(feature_range=(0,1)) #This gives us values between 0 and 1.
rescaledX = scaler.fit_transform(x) #crear matriz con método anterior
np.set_printoptions(precision=3) #setting precision for the output; núm de decimales
rescaledX[0:5, :] #ver primeras 5 filas de toda la matriz X



# Standardizing data
"""With standardizing, we can take attributes with a Gaussian distribution and different means 
and standard deviations and transform them into a standard Gaussian distribution with
 a mean of 0 and a standard deviation of 1. """

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# una vez cargado los datos, eliminando null y transformando variables catégoricas a númeircas:
base = pd.read_csv("C:/Users/eduar/OneDrive/Escritorio/Ejercicios Python/datos_ejercicios.csv")
base1 = base[["ANIO_GASTO", "MES_GASTO", "Edad", "Gasto",
              "Tipo_empleado"]]
base1["Tipo_empleado"] = base1["Tipo_empleado"].apply(lambda x: 1 if x=="Activo" else 0)
base2 = base1.dropna()
array = base2.values #valores de un DataFrame como un objeto NumPy array.
array

x = array[:,0:4 ] #crear x, con valores de las primeras 3 columnas de la matriz
y = array[:, 4] #crear y, con los valores de la columna 4
   
    
#crear objeto para transformar
scaler = StandardScaler().fit(x)
rescaledX = scaler.transform(x)
rescaledX[0:7, :]


#Normalizing data
"""In this task, we rescale each observation to a length of 1
(a unit norm). For this, we use the Normalizer class"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer


# una vez cargado los datos, eliminando null y transformando variables catégoricas a númeircas:
base = pd.read_csv("C:/Users/eduar/OneDrive/Escritorio/Ejercicios Python/datos_ejercicios.csv")
base1 = base[["ANIO_GASTO", "MES_GASTO", "Edad", "Gasto",
              "Tipo_empleado"]]
base1["Tipo_empleado"] = base1["Tipo_empleado"].apply(lambda x: 1 if x=="Activo" else 0)
base2 = base1.dropna()
array = base2.values #valores de un DataFrame como un objeto NumPy array.
array

x = array[:,0:4 ] #crear x, con valores de las primeras 3 columnas de la matriz
y = array[:, 4] #crear y, con los valores de la columna 4

scaler = Normalizer().fit(x)
normalizedX = scaler.transform(x)
normalizedX[0:8, :]


#Binarizing data
"""
Using a binary threshold, it is possible to transform our data by marking the values 
above it 1 and those equal to or below it, 0. For this purpose, 
we use the Binarizer class.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer


# una vez cargado los datos, eliminando null y transformando variables catégoricas a númeircas:
base = pd.read_csv("C:/Users/eduar/OneDrive/Escritorio/Ejercicios Python/datos_ejercicios.csv")
base1 = base[["ANIO_GASTO", "MES_GASTO", "Edad", "Gasto",
              "Tipo_empleado"]]
base1["Tipo_empleado"] = base1["Tipo_empleado"].apply(lambda x: 1 if x=="Activo" else 0)
base2 = base1.dropna()
array = base2.values #valores de un DataFrame como un objeto NumPy array.
array

x = array[:,0:4 ] #crear x, con valores de las primeras 3 columnas de la matriz
y = array[:, 4] #crear y, con los valores de la columna 4

#crear objeto Binarizer y ajustarlo a los datos de x
binarizer = Binarizer(threshold=0.0).fit(x) #parámetro threshold especifica 
#el umbral que se usará para binarizar los datos; el umbral se establece en 0.0.

binaryX = binarizer.transform(x)
binaryX[0:9, :]


#Mean Removal
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale


# una vez cargado los datos, eliminando null y transformando variables catégoricas a númeircas:
base = pd.read_csv("C:/Users/eduar/OneDrive/Escritorio/Ejercicios Python/datos_ejercicios.csv")
base1 = base[["ANIO_GASTO", "MES_GASTO", "Edad", "Gasto",
              "Tipo_empleado"]]
base1["Tipo_empleado"] = base1["Tipo_empleado"].apply(lambda x: 1 if x=="Activo" else 0)
base2 = base1.dropna()
base2.head()



data_standardized = scale(base2)
data_standardized.mean(axis = 0) #fijar media en CERO
data_standardized #ver valores

data_standardized.std(axis=0) #desviación estándar de las columnas


#One hot encoding
"""
When dealing with few and scattered numerical values, we may not need to 
store these. Then, we can perform One Hot Encoding. For k distinct values, 
we can transform the feature into a k-dimensional vector with one 
value of 1 and 0 as the rest values.
"""
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
encoder.fit([[0,1,6,2],
[1,5,3,5],
[2,4,2,7],
[1,0,4,2]
])

#
encoder.transform([[2,4,3,4]]).toarray()


# Label encoding
"""
Some labels can be words or numbers. Usually, training data is labelled 
with words to make it readable. Label encoding converts word labels 
into numbers to let algorithms work on them.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# una vez cargado los datos, eliminando null y transformando variables catégoricas a númeircas:
base = pd.read_csv("C:/Users/eduar/OneDrive/Escritorio/Ejercicios Python/datos_ejercicios.csv")
base1 = base[["ANIO_GASTO", "Edad", "Gasto", "Genero",
              "Tipo_empleado"]]
base1.head()

base2 = base1.dropna()
base2["Tipo_empleado"].unique() 
base2["Genero"].unique()


#Label encoder
label_encoder = LabelEncoder()
input_classes = ["Femenino", "Masculino", "Hombre", "Mujer"]# enumearar 
label_encoder.fit(input_classes)

for i, item in enumerate(label_encoder.classes_):
    print(item, "----->", i)

"""
Femenino -----> 0
Hombre -----> 1
Masculino -----> 2
Mujer -----> 3
"""
#si aplicamos lo anterio a una etiqueta como la siguiente:
labels = ["Masculino", "Hombre", "Femenino", "Mujer"]
#label_encoder.transform(labels)
labels = label_encoder.transform(labels)

#reusltado es: array([2, 1, 0, 3])
labels    


continuar = "https://data-flair.training/blogs/python-ml-data-preprocessing/"
"""4. Analyzing Data in Python Machine Learning
Assuming that you have loaded your dataset using pandas (which if you haven’t, refer to Python Pandas Tutorial to learn how), let’s find out more about our data."""



















































































































































































































