#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:06:04 2021

@author: cbarhor
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


plt.style.use('seaborn')

class MoviePredictor():
    
    def __init__(self, path="./Data/movies.csv"):
        self.df = self.read_and_clean(path)
        self.preprocess()
        
    def read_and_clean(self,path):
        """
        Método empleado para la lectura del fichero csv.
        Posteriormente, se limpia el conjunto de datos

        Parameters
        ----------
        path : str
            directorio relativo donde se encuentra el fichero csv.

        Returns
        -------
        df : pandas dataFrame
            base de datos limpia y preparada para su uso.

        """
        #Se lee el fichero csv
        df = pd.read_csv(path)
        #Inspección básica del conjunto de datos
        print(df.head())
        print(df.shape)
        print(df.dtypes)
        
        
        #Se eliminan las columnas que no se van a emplear por estar excluídas del reto
        df.drop('director', axis=1, inplace=True)
        df.drop('star', axis=1, inplace=True)
        df.drop('writer', axis=1, inplace=True)  
        #En 2020 los datos sufren una anomalía por culpa de la pandemia, se descartan
        df = df[df.year != 2020]

        #Se eliminan las filas con campos vacíos
        df.dropna(axis=0, inplace = True)
        df.reset_index(drop=True, inplace =True)
        
        #Se extrae el mes de lanzamiento:
        dates = []
        for i in df['released']:
            date = i.split("(")[0]              
            dates.append(date)
            
        df['released_date'] = dates
        df['released_date'] = pd.to_datetime(df['released_date'])
        df['month'] = df['released_date'].dt.month       
        df.drop(['released_date'], axis=1, inplace=True)       
 
        #Se eliminan las filas duplicadas
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace =True)
        
        #Se convierten a tipo int las siguientes columnas
        df.gross = df.gross.astype(int)
        df.budget = df.budget.astype(int)
        df.votes = df.votes.astype(int)
        df.month = df.month.astype(int)

        #El conjunto de datos finalmente queda así
        df.reset_index(drop=True, inplace =True)
        print(df.describe().T)
        print(df.shape)
        return df

  
    def preprocess(self):
        """
        Método empleado para hacer una primera inspección estadística del
        dataFrame. Con él se tomarán decisiones:
            - Las variables más fuertemente correladas con gross (variable a maximizar) son:
                + Budget 
                + Votes
                + Runtime
                + Year
                + Score
            - Estas variables se emplearán como features para crear el modelo
            de regresión. Se excluye la variable year ya que no es un factor que
            se pueda contrlolar a la hora de producir una película.
            
            - Las nubes de puntos no se pueden agrupar (clusterizar) en general
            de una manera coherente, por lo que no es posible distinguir zonas
            que vayan a suponer un claro cambio para el modelo (si hubiera, por ejemplo,
            dos clusters en la nube de puntos gross-runtime siempre se seleccionaría la duración 
            que mayor beneficio pueda aportar).

        Returns
        -------
        None.

        """
        #Se comprueba la correlación entre las distintas columnas del dataFrame
        self.checkAllCorrelation()
        #Se calculan los pairplot de las columnas entre sí
        self.showPairPlot()
        
    def showPairPlot(self):
        df = self.df.copy()
        sns.pairplot(df,diag_kind='hist')
        plt.show()
        sns.pairplot(df,hue='gross')
        plt.show()


            
    def checkAllCorrelation(self):        
        df = self.df.copy()
        #Se transforman las variables categoricas a numéricas:
        for col in df.columns:
            if (df[col].dtype == 'object'):
                df[col] = df[col].astype('category')
                df[col] = df[col].cat.codes
                
        #Se calcula la autocorrelación de gross con el resto de columnas        
        #Se excluyen todas aquellas correlaciones menores de abs(0.05) y se ordenan de mayor a menor
        df_corr = pd.DataFrame(df.corr().gross[(abs(df.corr().gross)) > 0.05].sort_values(ascending=False))
        #Se excluye la autocorrelación de gross
        df_corr = df_corr.iloc[1:].reset_index() 
        
    def useLinearRegression(self):
        """
        Método empleado a modo de pipeline para probar varios modelos
        de regresión lineal con diferentes parámetros

        Returns
        -------
        None.

        """
        #En este modelo se emplean las features más correladas, mencionadas anteriormente
        #Su r-squared es del 69.97%.
        #La r-squared es una métrica que indica cuánto de capaz es el modelo de
        #predecir una variable ciega en función de las features dadas. En otras
        #palabras, cómo de bueno es el modelo o cuánta variabilidad puede explicar
        features= ['budget', 'votes', 'runtime', 'score']
        x_train, x_test, y_train, y_test, scaler = self.prepareDataset(self.df.copy(),features)
        linreg = self.trainModel(x_train, x_test, y_train, y_test)
        self.showModelPerf(linreg, x_test, y_test)


        #En este modelo se emplean las mismas features, pero se limita el periodo
        #temporal a peliculas estrenadas después de 1995
        #R-squared = 67.16%
        #Parece ser que la limitación de datos pasa factura al modelo y que las tendencias
        #no han cambiado demasiado con los años (no tanto como para suponer ruido al modelo).
        #Empeora ligeramente respecto al modelo original
        #Se descarta
        df = self.df.copy()
        df_last_years = df[df.year > 1995]
        x_train, x_test, y_train, y_test, scaler = self.prepareDataset(df_last_years,features)
        linreg = self.trainModel(x_train, x_test, y_train, y_test)
        self.showModelPerf(linreg, x_test, y_test) 



        #En este nuevo modelo se pretende simplificar la dimensionalidad de las features
        #Para ello, se emplea el algoritmo PCA.
        #Con esto se pretende, manteniendo la misma información, reducir el número
        #de datos que se usan para entrenar. Esto agiliza el entrenamiento, y en muchos casos,
        #facilita la convergencia.
        #Se emplea el dataset original, con todos los años
        #R-squared = 69.59%. Empeora ligeramente respecto al modelo original
        #Esto puede ser porque las features no se encuentran fuertemente correladas
        #y las nuevas dimensiones no aportan datos realemnte significativos.
        #Se descarta
        x_train, x_test, y_train, y_test = self.prepareDatasetPCA(self.df.copy(),features)
        linreg = self.trainModel(x_train, x_test, y_train, y_test)
        self.showModelPerf(linreg, x_test, y_test) 



        #Este nuevo modelo hace uso, además, de las varibles categorícas más correladas.
        #Se espera que el aumento de datos facilite la convergencia.
        #Al tratarse de variables categoricas, hay que realizar un pequeño 
        #preprocesado antes.
        #Además, se crea un conjunto de características (point) que se piensa puede resultar
        #en una película con una buena taquilla. Este dato se utilizará como input para el modelo
        #ya creado y entrenanado.
        #R-Squared=70.46% Es el modelo con mejores resultados. Se selecciona.
        #Las predicciones indican que, con estos datos, la pelicula conseguirá un ratio de
        #cuatro veces mas beneficio que inversión.
        #Por ello, la película deberá tener un presupuesto de 150M$, 600.000 votos, 
        #una duración de 125 min, u IMDb = 7.7, ser una película familiar para todos los públicos
        #y estrenarse en junio.
        features= ['budget', 'votes', 'runtime', 'score', 'rating','genre','month','gross']
        df_num = self.df[features].copy()

        point = [150000000,600000,125,7.7,'G','Family', 6]
        #Se incluye un falso dato de gross para poder incluir el punto en el dataFrame
        point.append(0.0)
        df_num.loc[df_num.shape[0]] = (point)
        #Se transforman las variables categoricas de todo el dataFrame
        for col in df_num.columns:
            if (df_num[col].dtype == 'object'):
                df_num[col] = df_num[col].astype('category')
                df_num[col] = df_num[col].cat.codes   
        #Una vez transformado, se extrae el punto del conjunto del dataFrame
        test = [df_num.iloc[-1].drop('gross')]
        df_num.drop(df_num.tail(1).index, axis=0, inplace=True)
        #Se elimina 'gross' del conjunto de características (es la variable a predecir)
        features.remove('gross')  
        x_train, x_test, y_train, y_test, scaler = self.prepareDataset(df_num,features)
        #Se reescala el punto de test según se ha hecho con el resto de los datos
        test = scaler.transform(test)  
        linreg = self.trainModel(x_train, x_test, y_train, y_test)
        self.showModelPerf(linreg, x_test, y_test) 
        self.showCoefs(features,linreg)
        #Se realiza la predicción del 'gross' que generará nuestro point de test
        predicted_gross = linreg.predict(test)
        budget = point[0]
        #Resultados del modelo empleado
        print('Predicted gross: '+str(predicted_gross[0]))
        print('Ratio gross/budget: '+str(predicted_gross[0]/budget))
        

        
     
        
        
    def prepareDataset(self, df, features):
        """
        Método empleado para preparar y dividir
        el conjunto de datos en train y test.
        Los datos también se escalan y se preparan los labels

        Parameters
        ----------
        df : dataFrame
            dataFrame empleado en cada uno de los modelos.
        features : list
            Vector con los nombres de las features seleccionadas.

        Returns
        -------
        x_train : list
            Conjunto de datos de entrenamiento.
        x_test : list
            Etiquetas de datos de entrenamiento.
        y_train : list
            Conjunto de datos de test.
        y_test : list
            tiquetas de datos de test.
        scaler : object
            Objeto encargado de escalar de forma uniforme todo el dataset.

        """   
        #Selección de características dentro del dataFrame
        x = df[features]
        #Selección de la variable a predecir dentro dle dataFrame
        y = df.gross
        #Se separan los datos en train (67%) y test (33%)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
        #Se escalan los datos para que tengas diferencias equiparables entre sí (max y min entre 0 y 1, normalmente)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)  
        
        return x_train, x_test, y_train, y_test, scaler
    
    def prepareDatasetPCA(self, df, features):
        """
        Método empleado para preparar y dividir
        el conjunto de datos en train y test.
        Los datos también se escalan y se preparan los labels.
        Como principal novedad, se aplica el algoritmo PCA para intentar reducir las dimensiones de
        las features en 2

        Parameters
        ----------
        df : dataFrame
            dataFrame empleado en cada uno de los modelos.
        features : list
            Vector con los nombres de las features seleccionadas.

        Returns
        -------
        x_train : list
            Conjunto de datos de entrenamiento.
        x_test : list
            Etiquetas de datos de entrenamiento.
        y_train : list
            Conjunto de datos de test.
        y_test : list
            tiquetas de datos de test.

        """          
        x = df[features]
        y = df.gross
        
        #Se aplica el algoritmo PCA
        pca = PCA(n_components=len(features)-2)
        x = pca.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)  
        
        return x_train, x_test, y_train, y_test        
        
        
   
        
        
    def trainModel(self,x_train, x_test, y_train, y_test):
        """
        Método empleado para entrenar modelos 
        de regresión lineal según los datos aportados

        Parameters
        ----------
        x_train : list
            Conjunto de datos de entrenamiento.
        x_test : list
            Etiquetas de datos de entrenamiento.
        y_train : list
            Conjunto de datos de test.
        y_test : list
            tiquetas de datos de test.

        Returns
        -------
        linreg : object
            Objeto que contiene el modelo entrenado y listo para realizar predicciones.

        """
        #Se entrena el modelo
        linreg = LinearRegression()
        linreg.fit(x_train, y_train)
        #Se prueba el modelo
        predictions = linreg.predict(x_test)
        rsqrt = r2_score(y_test, predictions) * 100
        
        print('R-squared: {} %'.format(round(rsqrt,3)))
        return linreg
        
    def showModelPerf(self,linreg, x_test, y_test):
        """
        Método empleado para graficar las predicciones del modelo vs
        los valores reales de test. Con esta gráfica es fácil comprobar de forma
        visual cómo de bien actúa el modelo.

        Parameters
        ----------
        linreg : object
            Objeto que contiene el modelo entrenado y listo para realizar predicciones.
        y_train : list
            Conjunto de datos de test.
        y_test : list
            tiquetas de datos de test.

        Returns
        -------
        None.

        """
        predictions = linreg.predict(x_test)
        rsqrt = r2_score(y_test, predictions) * 100
        
        ax = sns.regplot(x=y_test, y=predictions,scatter_kws={'color': 'blue'}, line_kws={'color':'red'});
        ax.set(xlabel='Taquilla test', ylabel='Taquilla predicha');
        ax.set(title='Modelo de predicción: R-Squared= {}%'.format(round(rsqrt,3)));
        plt.show()
        
    def showCoefs(self,features,linreg):
        """
        Método empleado para mostrar los coeficientes del modelo entrenado.
        Estos pueden dar un idea de cómo se relaciona la variable a predecir en
        función de las features. Un coeficiente positivo indica que, aprox,
        las variables ciega e independiente crecen de forma proporcional.
        Un coeficiente negativo indica que las variables crecen de forma
        inversamente proporcional.

        Parameters
        ----------
        features : list
            Lista con los nombres de las características a emplear.
        linreg : object
            Objeto que contiene el modelo entrenado y listo para realizar predicciones.

        Returns
        -------
        None.

        """
        coeff = pd.DataFrame(index=features, columns=['Coefficients'], data=linreg.coef_)
        print(coeff)
        print("\n")
        
        
        
        
        
        
        
    def useKmeans(self, features=['gross','budget'],n_clusters = 2):
        """
        Método empleado para realizar la clusterización de variables, si fuese necesaria
        En este caso, no resulta útil intentar agrupar pares de variables. Sin
        embargo, se deja este método por si fuera útil en futuras implementaciones
        u otros datasets.
        Para realizar la clusterización se emplea el algoritmo no supervisado
        K-Means.

        Parameters
        ----------
        features : list, optional
            Nombres de las variables a comparar dentro del dataFrame. The default is ['gross','budget'].
        n_clusters : int, optional
            Número de clusters a emplear por K-Means. The default is 2.            

        Returns
        -------
        None.

        """
        
        df = self.df.copy()
        df_features = df[features].copy()
        model = self.trainKmeans(df_features, n_clusters)
        self.predictKmeans(model,df_features, features)
        
    def trainKmeans(self, data, n_clusters):
        """
        Método empleado para entrenar el algoritmo K-Means

        Parameters
        ----------
        data : dataFrame
            DataFrame con las features seleccionadas.
        n_clusters : int
            número de clusters a emplear.

        Returns
        -------
        model : object
            Objeto que contiene el modelo K-Means entrenado.

        """
        model = KMeans(n_clusters=n_clusters)      
        model.fit(data)
        return model
        
    def predictKmeans(self,model,  df, features):
        """
        Método empleado para realizar las agrupaciones con K-Means
        Muestra un gráfico con los resultados por pantalla

        Parameters
        ----------
        model : object
            Objeto que contiene el modelo K-Means entrenado.
        data : dataFrame
            DataFrame con las features seleccionadas.
        features : list
            Nombres de las variables a comparar dentro del dataFrame.

        Returns
        -------
        None.

        """

        all_predictions = model.predict(df)
        plt.scatter(x = df[features[0]], y=df[features[1]], c=all_predictions, edgecolor='red', s=40, alpha = 0.5)
        plt.xlabel(str(features[0]))
        plt.ylabel(str(features[1]))
        plt.show()
                
                
          
if __name__ == "__main__":
    moviePredictor = MoviePredictor()  
    moviePredictor.useKmeans()     
    moviePredictor.useLinearRegression()  

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        