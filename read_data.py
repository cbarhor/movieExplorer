#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:35:16 2021

@author: cbarhor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.style.use('seaborn')


class MovieExplorer():
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
        print(df.head())
        print(df.shape)
        print(df.dtypes)
        
        
        #Se eliminan las columnas que no se van a emplear por estar excluídas del reto
        df.drop('director', axis=1, inplace=True)
        df.drop('star', axis=1, inplace=True)
        df.drop('writer', axis=1, inplace=True)  
        
        #Se comprueba el porcentaje de datos que faltan por columna
        for cols in df.columns:
            pct_missing = np.mean(df[cols].isnull())*100
            print('{} - {}%'.format(cols, round(pct_missing,2)))
            
        
        #Se imprime por pantalla el nombre de todas las columnas
        print(df.columns)
        
        #Se eliminan aquellas filas que tengan alguna columna sin valor establecido
        data_before = df.shape[0]
        df.dropna(axis=0, inplace = True)
        df.reset_index(drop=True, inplace =True)        
        dropped = data_before - df.shape[0]
        print("dropped: "+str(dropped) + ", "+str(round((dropped/data_before)*100,2))+"%")
        print("total: "+str(df.shape[0]))  
        
        #Se extraen el mes de lanzamiento y el país de estreno
        dates = []
        rel_countries = []
        for i in df['released']:
            date, rel_country = i.split("(")
            rel_country = rel_country.split(")")[0]
              
            dates.append(date)
            rel_countries.append(rel_country)
                      
        df['released_date'] = dates
        df['released_date'] = pd.to_datetime(df['released_date'])
        df['month'] = df['released_date'].dt.month       
        df.drop(['released_date'], axis=1, inplace=True)       
        df['rel_country'] = rel_countries
        
        #Se genera una nueva columna con la diferencia entre
        #la taquilla y el presupuesto, es decir, con los 
        #beneficios netos
        df['income']= df['gross'] - df['budget']
        #Se eliminan las filas duplicadas
        df.drop_duplicates(inplace=True)
        #Se transforman a int los valores de las siguientes columnas
        df.gross = df.gross.astype(int)
        df.budget = df.budget.astype(int)
        df.votes = df.votes.astype(int)
        #Se genera una nueva columna con el ratio taquilla/budget de cada película
        ratio = df['gross']/df['budget']
        df['ratio'] = ratio
        df.ratio = df.ratio.astype(int)
        
        #El dataFrame queda de la siguiente forma
        df.reset_index(drop=True, inplace =True)
        print(df.describe().T)
        print(df.shape)
        return df

    def checkGauss(self, df, variable_name):
        """
        Método empleado para comprobar si los datos de la columna pasada como variable
        se asemejan a una distribución gaussiana o no.
        Para ello se emplea el test Kolmogorov-Smirnov de una sola variable
        
        Si el parámetro p resulta ser menor de 0.05, entonces el test arroja
        que la variable bajo estudio no se asemeja a la distribución con la que
        se está comparando.
        
        Se ha comprobado, según los resultados obtenidos, ninguna variable posee
        una distribución gaussiana

        Parameters
        ----------
        df : dataFrame
            conjunto de datos al completo.
        variable_name : str
            Nombre de la columna del dataFrame que se desea analizar.

        Returns
        -------
        None.

        """
        print("KS Test for {}: ".format(variable_name))
        x = df[str(variable_name)]
        ks_test = stats.kstest(x, 'norm')
        print(ks_test)

    
    def plotDist(self, df, variable_name):
        """
        Método empleado para mostrar la distribución de la variable
        seleccionada
        
        Haciendo una inspección visual se puede observar que las únicas
        variables que se asemejan a una distribución variable son:
            - Runtime
            - Score

        Parameters
        ----------
        df : dataFrame
            conjunto de datos al completo.
        variable_name : str
            Nombre de la columna del dataFrame que se desea analizar.

        Returns
        -------
        None.

        """
        # fig = plt.figure(figsize=(25,10))        
        sns.displot(df[str(variable_name)],bins=12)
        plt.title('Distribución de la variable {}'.format(variable_name), loc='left', fontsize=18, pad=20)
        plt.show()        
        
    def preprocess(self):
        """
        Método empleado para realizar un preporcesado básico sobre el dataset
        y poder conocer, en una primera aproximación, sus propiedades.
        Se estudia la gaussianidad de las distribuciones y la correlación entre
        las diferentes variables

        Returns
        -------
        None.

        """
        df = self.df.copy()
        variables_to_inspect = ['budget','gross','score','runtime', 'month','year']
        for variable_name in variables_to_inspect:
            self.checkGauss(df, variable_name)
            self.plotDist(df, variable_name)
            
        self.checkNumericCorrelation()
        self.checkAllCorrelation()
        self.showPairPlot() 
        
    def showPairPlot(self):
        """
        Método empleado para mostrar las relaciones entre las variables del
        dataFrame

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.pairplot(df,diag_kind='hist')
        plt.show()
        
    def checkNumericCorrelation(self):
        """
        Método empleado para estudiar la correlación entre las variables 
        numéricas del conjunto de datos bajo estudio
        Se muestra un mapa de calor con los resultados

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.heatmap(df.corr(method='pearson'), annot=True, vmin=-1, cmap='Blues')
        plt.title('Matriz de correlación (características numéricas)', loc='left', fontsize=18, pad=20)
        plt.show()
        
        

            
    def checkAllCorrelation(self):
        """
        Método empleado para estudiar la correlación entre las variables 
        numéricas y categóricas del conjunto de datos bajo estudio
        Se muestra un mapa de calor con los resultados

        Returns
        -------
        None.

        """        
        df = self.df.copy()
        #Se convierten las varibles categóricas a numéricas
        for col in df.columns:
            if (df[col].dtype == 'object'):
                df[col] = df[col].astype('category')
                df[col] = df[col].cat.codes
        
        df_corr = df.corr(method='pearson')
        sns.heatmap(df_corr, annot=True, vmin=-1, cmap='YlOrBr')
        plt.title('Matriz de correlación completa', loc='left', fontsize=18, pad=20)
        plt.show()
        #Se muestra por pantalla aquellas variables que poseen una correlación mayor que 0.5
        #Se excluyen autocorrelaciones
        corr_pairs = df_corr.unstack()
        corr_pairs = corr_pairs[corr_pairs<1.0]
        high_corr = corr_pairs[corr_pairs>0.5]               
        top_corr = high_corr.sort_values(ascending=False) 
        print(top_corr.head(10))
    
#%%
#Exploratory Data Analysis (EDA):
    def makeEDAnalysis(self):
       #Explore dataset:
        # By years (2020==pandemic):
        self.plotRatingvsYears()
        self.plotNumbervsYear()
        self.plotYearlyGrossPercentage()
        self.plotCountryvsNumberByYear()
        #By year & month:
        self.plotBestMothbyYear()    
        #By film:
        self.plotTop10GrossMovies()
        self.plotMoviesGrossandBudget()
        self.plotRemakevsGross()
        #By companies:
        self.plotCompaniesGrossandBudget()
        self.plotCompaniesGrossandGenre()
        self.plotCompaniesGrossandGenreMean()
        
        #By rating:
        self.plotRatings()  
        self.plotRatingByGenere()
        self.plotRatingvsGross()  
        self.plotRatingvsRatio()
        
        #By genre:
        self.plotNumbervsGenre()
        self.plotGenrevsNumber()
        self.plotGenrevsGross()
        self.plotGenrevsBudget()
        self.plotGenrevsGrossBudgetRatio()
        
        self.plotGrossvsGenreBox()
        self.plotBudgetvsGenreBox()
        self.plotGenrevsRatioBox()
        self.plotBestMothbyGenre()
        
        self.plotGenrevsYears()
        self.plotGenrevsNumberByYear()
        self.plotGenrevsIMDB()
        
        #By incomes (gross-budget):
        self.plotIncomes()
        
        #By score:
        self.plotGreaterIMDBvsGross()
        self.plotGreaterGrossvsIMDB()
        self.plotBudgetvsGrossvsScore()
        self.plotGrossvsIMDB()
        self.plotBudgetvsIMDB()
        self.plotIMDBvsYear()
        self.plotIMDBvsGenre()
        
        #By country:
        self.plotProductionCountryvsBudget()
        self.plotCountryvsGross()
        self.plotCountryvsNumber()
        self.plotReleaseCountryvsGross()
        
        #By runtime:
        self.plotRuntimevsGross()
        self.plotRuntimevsBudget()    
    
    def plotGreaterIMDBvsGross(self):
        """
        Método empleado para graficar la taquilla de las películas 
        mejor puntuadas (IMDb)
        Returns
        -------
        None.

        """
        df = self.df.copy()
        best_score = df.sort_values('score',ascending=False).head(7)
        sns.barplot(x='name', y ='gross',hue='score', data=best_score)   
        plt.xticks(rotation=90)    
        plt.xlabel('Película')
        plt.ylabel('Taquilla')
        plt.title('Taquilla de las películas mejor puntuadas', loc='left', fontsize=18, pad=20)
        plt.show()

    def plotGreaterGrossvsIMDB(self):
        """
        Método empleado para graficar la puntuación IMDb de las películas más
        taquilleras del dataset

        Returns
        -------
        None.

        """
        df = self.df.copy()
        best_gross = df.sort_values('gross',ascending=False).head(7)
        sns.barplot(x='name', y ='gross',hue='score', data=best_gross)       
        plt.xticks(rotation=90) 
        plt.xlabel('Película')
        plt.ylabel('Taquilla')
        plt.title('Puntuación de las películas más taquilleras', loc='left', fontsize=18, pad=20)
        plt.show()

    def plotCompaniesGrossandBudget(self):
        """
        Método empleado para graficar el presupuesto y la taquilla totales
        de las compañías más taquilleras

        Returns
        -------
        None.

        """
        df = self.df.copy()
        companies = df.groupby('company')['gross','budget'].sum().reset_index().sort_values('gross',ascending=False).head(10)
        companies.plot(x='company',y=['budget','gross'],kind='bar',figsize=(12,5))
        plt.title("Taquilla y presupuesto por compañía",size=30)
        plt.xlabel("Compañía",size=15)
        plt.ylabel("Cantidad",size=15)
        plt.show()
        



        
        
    def plotCompaniesGrossandGenre(self):
        """
        Método empleado para graficar la taquilla total y la cantidad que le 
        pertenece a cada género

        Returns
        -------
        None.

        """
        df = self.df.copy()                
        pivot_movie = df.groupby(['company','genre'], as_index=False)['gross'].sum().pivot(index='company',columns='genre', values='gross').fillna(0)
        gross_sum = pivot_movie.sum(axis=1)
        pivot_movie['gross_sum'] = gross_sum
        top_gross_company = pivot_movie.sort_values(by='gross_sum', ascending = False).head(10)
        top_gross_company.sort_values(by='gross_sum', ascending =True, inplace=True)
        top_gross_company.drop('gross_sum', axis=1, inplace=True)
        
        top_gross_company.plot(kind='barh', stacked=True, figsize=(30,15))
        plt.xlabel("Taquilla", fontsize=20, labelpad=10)
        plt.ylabel("Compañia", fontsize=20, labelpad=10)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(10^3,10^3))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Top 10 compañías según taquilla', pad=20, fontsize=26)
        plt.legend(fontsize=16)
        plt.show()




    def plotCompaniesGrossandGenreMean(self):
        """
        Método empleado para graficar la taquilla media y la cantidad media que le 
        pertenece a cada género

        Returns
        -------
        None.

        """
        df = self.df.copy()               
        pivot_movie = df.groupby(['company','genre'], as_index=False)['gross'].median().pivot(index='company',columns='genre', values='gross').fillna(0)
        gross_sum = pivot_movie.sum(axis=1)
        pivot_movie['gross_sum'] = gross_sum
        top_gross_company = pivot_movie.sort_values(by='gross_sum', ascending = False).head(10)
        top_gross_company.sort_values(by='gross_sum', ascending =True, inplace=True)
        top_gross_company.drop('gross_sum', axis=1, inplace=True)
        
        top_gross_company.plot(kind='barh', stacked=True, figsize=(30,15))
        plt.xlabel("Taquilla", fontsize=20, labelpad=10)
        plt.ylabel("Compañia", fontsize=20, labelpad=10)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(10^3,10^3))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Taquilla media de cada compañía según género', pad=20, fontsize=26)
        plt.legend(fontsize=16)
        plt.show()


    def plotRatingByGenere(self):
        """
        Método empleado para graficar la cantidad de películas que hay en el dataset 
        distribuidas por género y clasificación de edad

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.countplot(x = 'rating',data = df , hue='genre')
        plt.xlabel("Clasificación (edad)", fontsize=20, labelpad=10)
        plt.xticks(fontsize=16,rotation=90)
        plt.yticks(fontsize=16)
        plt.title('Cantidad de películas por clasificación y género', pad=20, fontsize=26)        
        plt.legend(loc='upper center')
        plt.show()


    def plotMoviesGrossandBudget(self):
        """
        Método empleado para graficar la taquilla y el presupesto empleado
        en las películas con mayor recaudación

        Returns
        -------
        None.

        """
        df = self.df.copy()
        movies = df.groupby('name')['gross','budget'].sum().reset_index().sort_values('gross',ascending=False).head(10)
        movies.plot(x='name',y=['budget','gross'],kind='bar',figsize=(12,5))
        plt.title("Taquilla y presupuesto por película",size=30)
        plt.xlabel("Película",size=15)
        plt.ylabel("Cantidad",size=15)
        plt.show()
    def plotNumbervsGenre(self):
        """
        Método empleado para graficar la relación entre la cantidad de pel´culas de
        cada género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        df_genre = df.groupby(['genre']).size().sort_values(ascending=False).reset_index()
        df_genre.columns = ['genre', 'count']
        others = df_genre.loc[list(range(8,15))]['count'].sum()
        df_genre = df_genre.drop(list(range(8,15)))
        df_genre.loc[8] = ['Others', others]
        
        labels = df_genre['genre']
        size = df_genre['count']
        plt.pie(size, labels = labels, explode = [.1,0,0,0,0,0,0,0,0], autopct='%1.1f%%', shadow = True)
        plt.title('Películas según género', pad=20, fontsize=26) 
        plt.show()


    def plotBudgetvsGrossvsScore(self):
        """
        Método empleado para graficar la taquilla y el presupuesto de cada
        película. Además, se incluye un mapa de calor con la puntuación IMDb

        Returns
        -------
        None.

        """

        df = self.df.copy()
        plt.scatter(x=df['budget'],y=df['gross'], c=df['score'], cmap="plasma_r")
        plt.xlabel('Presupuesto', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Taquilla vs Presupuesto', size=20, pad=10)
        cbar = plt.colorbar()
        cbar.set_label('IMDb', labelpad=10, size=14)
        plt.show()

    # def processGenre(self):
    #     df = self.df.copy()
    #     movie_genre_gross = df.groupby(['genre'])['gross'].median().to_frame().reset_index().sort_values('gross', ascending = False).head(19)
    #     movie_genre_gross['genre'].to_list()
    #     print(movie_genre_gross)        
    #     df = self.df.copy()
    #     movie_genre_IMDB = df.groupby(['genre'])['score'].median().to_frame().reset_index().sort_values('score', ascending = False).head(19)
    #     movie_genre_IMDB['genre'].to_list()
    #     print(movie_genre_IMDB)
                

    def plotGenrevsGross(self):
        """
        Método empleado para graficar la taquilla media que recauda cada
        película según su género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))
        genre_vs_gross = df.groupby(['genre'])['gross'].median().reset_index().sort_values('gross', ascending = False).head(8)
        sns.barplot(x='genre', y ='gross', data=genre_vs_gross)
        plt.xlabel('Género')
        plt.ylabel('Taquilla')
        plt.title('Taquilla media por género', loc='left', fontsize=18, pad=20)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()
        
    def plotRatings(self):
        """
        Método empleado para graficar la relación del número de películas según
        su clasificación de edad

        Returns
        -------
        None.

        """
        df = self.df.copy()
        df_genre = df.groupby(['rating']).size().sort_values(ascending=False).reset_index()
        df_genre.columns = ['rating', 'count']
        others = df_genre.loc[list(range(4,10))]['count'].sum()
        df_genre = df_genre.drop(list(range(4,10)))
        df_genre.loc[4] = ['Others', others]       
        labels = df_genre['rating']
        size = df_genre['count']
        plt.pie(size, labels = labels, explode = [.1,0,0,0,0], autopct='%1.1f%%', shadow = True)
        plt.title('Películas según clasificación de edad', pad=20, fontsize=26) 
        plt.show()
        
    def plotRatingvsGross(self): 
        """
        Método empleado para graficar la taquilla recaudada según la clasificación 
        de edad

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))
        rating_df = df.sort_values('gross',ascending=False)
        ax = sns.boxplot(x='rating', y='gross', data=rating_df, showfliers=False)
        ax.tick_params(axis='both', labelsize=16)       
        ax.set_title('Taquilla según clasificación', fontsize=24, pad=10)
        ax.set_ylabel('Taquilla', fontsize=20)
        ax.set_xlabel('Clasificación (edad)', fontsize=20)  
        plt.show()
        
    def plotRatingvsRatio(self): 
        """
        Método empleado para graficar la relación entre la taquilla recaudada
        y el presupuesto de cada película, distribuido según la clasifiación de edad

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))
        rating_df = df.sort_values('ratio',ascending=False)
        ax = sns.boxplot(x='rating', y='ratio', data=rating_df, showfliers=False)
        ax.tick_params(axis='both', labelsize=16)       
        ax.set_title('Ratio Taquilla/Presupuesto según clasificación', fontsize=24, pad=10)
        ax.set_ylabel('Ratio Taquilla/Presupuesto', fontsize=20)
        ax.set_xlabel('Clasificación (edad)', fontsize=20)     
        plt.show()
        
    def plotGenrevsBudget(self):
        """
        Método empleado para graficar el presupuesto medio de cada película
        según su género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))
        genre_vs_budget = df.groupby(['genre'])['budget'].median().reset_index().sort_values('budget', ascending = False).head(8)
        sns.barplot(x='genre', y ='budget', data=genre_vs_budget)
        plt.xlabel('Género')
        plt.ylabel('Presupuesto')
        plt.title('Presupuesto medio según género', loc='left', fontsize=18, pad=20)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))        
        plt.show()
        
    def plotGenrevsGrossBudgetRatio(self):
        """
        Método empleado para graficar la relación media entre la taquilla recaudada
        y el presupuesto de cada película según su género. Barplot

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))       
        mean_ratio_df = df.groupby(['genre'])['ratio'].median().to_frame().reset_index().sort_values('ratio',ascending=False).head(10)
        sns.barplot(x='genre', y ='ratio', data=mean_ratio_df)
        plt.xlabel('Género')
        plt.ylabel('Ratio Taquilla/Presupuesto')
        plt.title('Ratio Taquilla/Presupuesto según género', loc='left', fontsize=18, pad=20)
        plt.show()
        
    def plotGenrevsRatioBox(self):
        """
        Método empleado para graficar la relación media entre la taquilla recaudada
        y el presupuesto de cada película según su género. Boxplot

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))
        ratio_df = df.sort_values('ratio',ascending=False)
        ax = sns.boxplot(x='genre', y='ratio', data=ratio_df, showfliers=False)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title('Ratio Taquilla/Presupuesto según género', fontsize=24, pad=10)
        ax.set_ylabel('Ratio Taquilla/Presupuesto', fontsize=20)
        ax.set_xlabel('Género', fontsize=20) 
        plt.show()
        
    def plotGenrevsYears(self):
        """
        Método empleado para graficar la tendencia en taquilla con los años de
        las películas según su género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        movie_genre_gross_by_year = df.groupby(['year','genre'])['gross'].median().unstack().fillna(0)
        
        plt.figure(figsize=(25,10))
        ax = sns.lineplot(data=movie_genre_gross_by_year, dashes=False, lw=3, palette='Set2')
        leg = ax.legend(fontsize=18)
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title('Tendencia en taquilla según género', fontsize=24, pad=10)
        ax.set_ylabel('Taquilla', fontsize=20, labelpad=10)
        ax.set_xlabel('Año', fontsize=20, labelpad=10)
        plt.show()

    def plotRatingvsYears(self):
        """
        Método empleado para graficar la tendencia por años en taquilla de las
        películas según su clasificación por edad

        Returns
        -------
        None.

        """
        df = self.df.copy()
        movie_genre_gross_by_year = df.groupby(['year','rating'])['gross'].median().unstack().fillna(0)
        plt.figure(figsize=(25,10))
        ax = sns.lineplot(data=movie_genre_gross_by_year, dashes=False, lw=3, palette='Set2')
        leg = ax.legend(fontsize=18)
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title('Tendencia en taquilla según clasificación', fontsize=24, pad=10)
        ax.set_ylabel('Taquilla', fontsize=20, labelpad=10)
        ax.set_xlabel('Año', fontsize=20, labelpad=10)
        plt.show()
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        

    def plotGenrevsIMDB(self):
        """
        Método empleado para graficar lapuntuación IMDb según el género de las
        películas

        Returns
        -------
        None.

        """
        plt.figure(figsize=(25,10))
        ax = sns.histplot(self.df, x="genre", y="score",multiple='stack', bins=19,  hue='genre')
        ax.set_title('IMDb según género', fontsize=24, pad=10)
        ax.set_ylabel('IMDb', fontsize=20, labelpad=10)
        ax.set_xlabel('Género', fontsize=20, labelpad=10)
        plt.show()
        
    def plotIMDBvsYear(self):
        """
        Método empleado para graficar la tendencia con los años en la puntuación
        media de las películas según su género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        imdb_by_year_and_genre = df.groupby(['year','genre'])['score'].median().unstack().fillna(method="ffill")        
        plt.figure(figsize=(25,10))
        ax = sns.lineplot(data=imdb_by_year_and_genre, dashes=False, lw=3, palette='Set2')
        leg = ax.legend(fontsize=18)
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title('Tendencia IMDb según género', fontsize=24, pad=10)
        ax.set_ylabel('IMDB', fontsize=20, labelpad=10)
        ax.set_xlabel('Año', fontsize=20, labelpad=10)
        plt.show()
        
    def plotIMDBvsGenre(self):
        """
        Método empleado para graficar la distribución de la puntuación IMDb
        en función del género
        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))
        ax = sns.boxplot(x='genre', y='score', data=df, showfliers=False)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title('IMDb según género', fontsize=24, pad=10)
        ax.set_ylabel('IMDb', fontsize=20)
        ax.set_xlabel('Género', fontsize=20)
        plt.show()

    def plotGrossvsGenreBox(self):
        """
        Método empleado para graficar la distribución de la recaudación en taquilla
        según el género de las películas

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))
        ax = sns.boxplot(x='genre', y='gross', data=df, showfliers=False)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title('Taquilla según género', fontsize=24, pad=10)
        ax.set_ylabel('Taquilla', fontsize=20)
        ax.set_xlabel('Género', fontsize=20) 

    def plotBudgetvsGenreBox(self):
        """
        Método empleado para graficar el presupuesto medio de cada película
        según su género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(25,10))
        ax = sns.boxplot(x='genre', y='budget', data=df, showfliers=False)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title('Presupuesto medio según género', fontsize=24, pad=10)
        ax.set_ylabel('Presupuesto', fontsize=20)
        ax.set_xlabel('Género', fontsize=20)         
        
    # def getIncomesRanking(self):
    #     """
    #     Método empleado para graficar 

    #     Returns
    #     -------
    #     None.

    #     """
    #     df = self.df.copy()
    #     movie_incomes_by_genre = df.groupby(['genre'])['income'].mean().to_frame().reset_index().sort_values('income', ascending = False).head(19)
    #     movie_incomes_by_genre['genre'].to_list()
    #     print(movie_incomes_by_genre)
    #     # self.plotIncomes()
    
        
    def plotIncomes(self):
        """
        Método empleado para graficar la tendencia con los años del beneficio neto
        (taquilla - presupuesto) medio en función del género de la película

        Returns
        -------
        None.

        """
        df = self.df.copy()
        movie_genre_gross_by_year = df.groupby(['year','genre'])['income'].mean().unstack().fillna(0)
        plt.figure(figsize=(25,10))
        ax = sns.lineplot(data=movie_genre_gross_by_year, dashes=False, lw=3, palette='Set2')
        leg = ax.legend(fontsize=18)
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title('Tendencia de beneficio neto según el género', fontsize=24, pad=10)
        ax.set_ylabel('Beneficio', fontsize=20, labelpad=10)
        ax.set_xlabel('Año', fontsize=20, labelpad=10)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()
        

    # def getBestMonthByGenre(self):
        
    #     df = self.df.copy()
    #     movie_gross_by_month = df.groupby(['month','genre'])['gross'].median().to_frame().reset_index().sort_values('gross', ascending = False).head(19)
    #     movie_gross_by_month['genre'].to_list()
    #     print(movie_gross_by_month)
    #     # self.plotBestMothbyGenre()

    def plotBestMothbyGenre(self):
        """
        Método empleado para graficar  la taquilla media según el mes de estreno
        y el género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        movie_gross_by_month = df.groupby(['month','genre'])['gross'].median().unstack().fillna(0)
        plt.figure(figsize=(10,8))
        ax = sns.heatmap(movie_gross_by_month, cbar_kws={'label': 'Taquilla'})
        ax.set_title('Taquilla según género y mes de estreno', fontsize=20, pad=10)
        ax.set_xlabel('Género', fontsize=16, labelpad=10)
        ax.set_ylabel('Mes', fontsize=16, labelpad=10)
        ax.tick_params(axis='both', labelsize=10)
        plt.show()
        
    def plotBestMothbyYear(self):
        """
        Método empleado para graficar la taquilla media de cada película según
        mes y año de estreno

        Returns
        -------
        None.

        """
        df = self.df.copy()
        movie_gross_by_month = df.groupby(['month','year'])['gross'].median().unstack().fillna(0)
        plt.figure(figsize=(10,8))
        ax = sns.heatmap(movie_gross_by_month, cbar_kws={'label': 'Taquilla'})
        ax.set_title('Taquilla según año y mes de estreno', fontsize=20, pad=10)
        ax.set_xlabel('Año', fontsize=16, labelpad=10)
        ax.set_ylabel('Mes', fontsize=16, labelpad=10)
        ax.tick_params(axis='both', labelsize=10)
        plt.show()   
        
    def plotTop10GrossMovies(self):
        """
        Método empleado para graficar la recaudación del top de películas 
        más taquilleras

        Returns
        -------
        None.

        """
        df = self.df.copy()
        df = df.sort_values(by=['gross'],ascending=False).head(10)
        sns.barplot(x='name', y ='gross', data=df)
        plt.xticks(rotation=90)
        plt.xlabel('Película')
        plt.ylabel('Taquilla')
        plt.title('Top 10 de películas más taquilleras', loc='left', fontsize=18, pad=20)
        plt.show()

    def plotRemakevsGross(self):
        """
        Método empleado para graficar la taquilla de las películas originales
        y de sus remakes

        Returns
        -------
        None.

        """
        df = self.df.copy()
        df = df.sort_values(by=['name','year'],ascending=False)
        df_original = df[df.name.duplicated(keep='first')].reset_index().sort_values(by=['name'],ascending=True).head(8)#.sort_values(by=['name','year'],ascending=True).head(12)
        df_remake = df[df.name.duplicated(keep='last')].reset_index().sort_values(by=['name'],ascending=True).head(8)
        df_remake['original_gross'] = df_original['gross']
        df_remake['original_year'] = df_original['year']       
        df_remake = df_remake.sort_values(by=['name'],ascending=True)        
        df_remake.plot(x='name',y=['original_gross','gross'], kind='bar',figsize=(12,5))
        plt.title("Taquilla por remake",size=30)
        plt.xticks(rotation=45)
        plt.xlabel("Película",size=15)
        plt.ylabel("Taquilla",size=15)
        plt.show()
        
       
        
    def plotGrossvsIMDB(self):
        """
        Método empleado para graficar la taquilla según la puntuación IMDb de 
        la película

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.scatter(y=df['gross'],x=df['score'])
        plt.ylabel('Taquilla', labelpad=10, size=14)
        plt.xlabel('IMDb', labelpad=10, size=14)
        plt.title('Taquilla según IMDb', size=20, pad=10)
        plt.show()


    def plotBudgetvsIMDB(self):
        """
        Método empleado para graficar el presupuesto de cada película en relación
        a su puntuación IMDb

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.scatter(y=df['budget'],x=df['score'])
        plt.ylabel('Presupuesto', labelpad=10, size=14)
        plt.xlabel('IMDb', labelpad=10, size=14)
        plt.title('Presupuesto según IMDb', size=20, pad=10)
        plt.show()
        
    def plotNumbervsYear(self):
        """
        Método empleado para graficar la cantidad de películas estrenadas 
        cada año

        Returns
        -------
        None.

        """
        df = self.df.copy()
        movies_per_year = df.groupby(['year']).count()
        plt.figure(figsize=(10,8))
        sns.lineplot(x=movies_per_year.index, y='name', data=movies_per_year, linewidth=4)
        plt.xticks(rotation=45)
        plt.xlabel('Año')
        plt.ylabel('Películas')
        plt.title('Cantidad de películas estrenadas por año', loc='left', fontsize=18, pad=20)
        plt.show()
        
    def plotCountryvsGross(self):
        """
        Método empleado para graficar la taquilla media de cada
        película según su país de preducción

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(10,8))
        country_mean_gross = df.groupby(['country'])['gross'].median().to_frame().reset_index().sort_values('gross', ascending = False).head(8)
        sns.barplot(x='country', y ='gross', data=country_mean_gross)
        plt.xticks(rotation=45)
        plt.xlabel('País de producción')
        plt.ylabel('Taquilla')
        plt.title('Taquilla media según el país de producción', loc='left', fontsize=18, pad=20)
        plt.show()
                
    def plotProductionCountryvsBudget(self):
        """
        Método empleado para graficar el presupuesto medio de cada película
        según su país de producción

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(10,8))
        country_mean_gross = df.groupby(['country'])['budget'].median().to_frame().reset_index().sort_values('budget', ascending = False).head(8)
        sns.barplot(x='country', y ='budget', data=country_mean_gross)
        plt.xticks(rotation=45)
        plt.xlabel('País de producción')
        plt.ylabel('Presupuesto')
        plt.title('Presupuesto medio según el país de producción', loc='left', fontsize=18, pad=20)
        plt.show()
                
    def plotReleaseCountryvsGross(self):
        """
        Método empleado para graficar la taquilla media de cada 
        película según su país de estreno

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(10,8))
        country_mean_gross = df.groupby(['rel_country'])['gross'].median().to_frame().reset_index().sort_values('gross', ascending = False).head(8)
        sns.barplot(x='rel_country', y ='gross', data=country_mean_gross)
        plt.xticks(rotation=45)
        plt.xlabel('País de estreno')
        plt.ylabel('Taquilla')
        plt.title('Taquilla media según el país de estreno', loc='left', fontsize=18, pad=20)
        plt.show()        
        
    def plotCountryvsNumber(self):
        """
        Método empleado para graficar la cantidad de películas producidas
        en cada país

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(10,8))
        country_production = df.groupby(['country']).count().sort_values('name', ascending = False).head(8)
        sns.barplot(x=country_production.index, y ='name', data=country_production)
        plt.xticks(rotation=45)
        plt.xlabel('País de producción')
        plt.ylabel('Películas')
        plt.title('Cantidad de películas producidas por país', loc='left', fontsize=18, pad=20)
        plt.show()



    def plotCountryvsNumberByYear(self):
        """
        Método empleado para graficar la cantidad media de películas producidas
        al año por cada país

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(10,8))
        country_production = df.groupby(['country','year']).count().groupby(['country'])['name'].median().to_frame().sort_values('name', ascending = False).head(8)
        sns.barplot(x=country_production.index, y = 'name', data=country_production)
        plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Películas')
        plt.title('Cantidad media de películas producidas al año por país', loc='left', fontsize=18, pad=20)
        plt.show() 




    def plotGenrevsNumber(self):
        """
        Método empleado para graficar la cantidad de películas de cada género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(10,8))
        country_production = df.groupby(['genre']).count().sort_values('name', ascending = False).head(8)
        sns.barplot(x=country_production.index, y ='name', data=country_production)
        plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Películas')
        plt.title('Cantidad de películas por género', loc='left', fontsize=18, pad=20)
        plt.show()                



    def plotGenrevsNumberByYear(self):
        """
        Método empleado para graficar la cantidad de películas producidas al
        año según su género

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.figure(figsize=(10,8))
        country_production = df.groupby(['genre','year']).count().groupby(['genre'])['name'].median().to_frame().sort_values('name', ascending = False).head(8)
        sns.barplot(x=country_production.index, y = 'name', data=country_production)
        plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Películas')
        plt.title('Cantidad media de películas por género y año', loc='left', fontsize=18, pad=20)
        plt.show() 
        
        
    def plotRuntimevsGross(self):
        """
        Método empleado para graficar la taquilla y el IMDb de cada película
        según su duración

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.scatter(x=df['runtime'],y=df['gross'], c=df['score'], cmap="plasma_r")
        plt.xlabel('Duración (en min)', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        plt.title('Taquilla según duración', size=20, pad=10)
        cbar = plt.colorbar()
        cbar.set_label('IMDb', labelpad=10, size=14)
        plt.show()

    def plotRuntimevsBudget(self):
        """
        Método empleado para graficar el presupuesto de cada película según
        su duración

        Returns
        -------
        None.

        """
        df = self.df.copy()
        plt.scatter(x=df['runtime'],y=df['budget'], c=df['score'], cmap="plasma_r")
        plt.xlabel('Duración (en min)', labelpad=10, size=14)
        plt.ylabel('Presupuesto', labelpad=10, size=14)
        plt.title('Presupuesto según duración', size=20, pad=10)
        cbar = plt.colorbar()
        cbar.set_label('IMDb', labelpad=10, size=14)
        plt.show()        
 
        
 
    def plotYearlyGrossPercentage(self):
        """
        Método empleado para graficar el porcentaje medio la taquilla de cada
        año

        Returns
        -------
        None.

        """
        df = self.df.copy()
        year_rev = df.groupby('year')['gross'].sum().pct_change().to_frame().reset_index().sort_values('year', ascending = True)
        year_rev_pct = year_rev.assign(year_pct = year_rev['gross']*100).reset_index()
        sns.barplot(x='year', y ='year_pct', data=year_rev_pct)
        plt.xticks(rotation=90)
        plt.xlabel('Año')
        plt.ylabel('Taquilla (%)')
        plt.title('Taquilla (%) por año', loc='left', fontsize=18, pad=20)
        plt.show()

     
        
#%%
# Correlation analysis:
    def makeCorrAnalysis(self):
        #Regression models:
        self.plotRuntimevsBudgetRegression()
        self.plotRuntimevsGrossRegression()
        self.plotBudgetvsGrossRegression()
        self.plotIMDBvsGrossRegression()
        self.plotIMDBvsBudgetRegression()
        self.plotVotesvsGrossRegression()                

        
    def plotRuntimevsGrossRegression(self):
        """
        Método empleado para graficar la correlación entre la duración de 
        cada película y su taquilla

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.regplot(x='runtime',y='gross',data=df,scatter_kws={"color":"blue"},line_kws={"color":"red"})
        plt.xlabel('Duración (en min)', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        plt.title('Taquilla según duración', size=20, pad=10)
        plt.show()

        
    def plotRuntimevsBudgetRegression(self):
        """
        Método empleado para graficar la correlación entre el presupuesto de cada película
        y su duración

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.regplot(x='runtime',y='budget',data=df,scatter_kws={"color":"red"},line_kws={"color":"green"})
        plt.xlabel('Duración (en min)', labelpad=10, size=14)
        plt.ylabel('Presupuesto', labelpad=10, size=14)
        plt.title('Presupuesto según duración', size=20, pad=10)
        plt.show()

    
    def plotBudgetvsGrossRegression(self):
        """
        Método empleado para graficar la correlación entre la taquilla de cada
        película y su presupuesto

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.regplot(x='budget',y='gross',data=df,scatter_kws={"color":"blue"},line_kws={"color":"red"})
        plt.xlabel('Presupuesto', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        plt.title('Taquilla según presupuesto', size=20, pad=10)
        plt.show()
        

    def plotVotesvsGrossRegression(self):
        """
        Método empleado para graficar la correlación entre la taquilla de una 
        película y sus votos

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.regplot(x='votes',y='gross',data=df,scatter_kws={"color":"blue"},line_kws={"color":"red"})
        plt.xlabel('Votos', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        plt.title('Taquilla según votos', size=20, pad=10)
        plt.show()

        
    def plotIMDBvsGrossRegression(self):
        """
        Método empleado para graficar la correlación entre la puntuación
        IMDb de cada película y su taquilla

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.regplot(x='gross',y='score',data=df,scatter_kws={"color":"blue"},line_kws={"color":"red"})
        plt.xlabel('Taquilla', labelpad=10, size=14)
        plt.ylabel('IMDb', labelpad=10, size=14)
        plt.title('IMDb según taquilla', size=20, pad=10)
        plt.show()

    def plotIMDBvsBudgetRegression(self):
        """
        Método empleado para graficar la correlación entre la puntuación
        IMDb de cada película y su presupuesto

        Returns
        -------
        None.

        """
        df = self.df.copy()
        sns.regplot(x='budget',y='score',data=df,scatter_kws={"color":"red"},line_kws={"color":"green"})
        plt.xlabel('Presupuesto', labelpad=10, size=14)
        plt.ylabel('IMDb', labelpad=10, size=14)
        plt.title('IMDb según presupuesto', size=20, pad=10)
        plt.show()
        



#%%
if __name__ == "__main__":
    #Read & clean data:
    movieExplorer = MovieExplorer()
    movieExplorer.makeEDAnalysis()
    movieExplorer.makeCorrAnalysis()
    
