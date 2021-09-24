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
        df = pd.read_csv(path)
        print(df.head())
        print(df.shape)
        print(df.dtypes)
        
        
        
        df.drop('director', axis=1, inplace=True)
        df.drop('star', axis=1, inplace=True)
        df.drop('writer', axis=1, inplace=True)  
        
        
        
        print(df.isna().sum())
        
        for cols in df.columns:
            pct_missing = np.mean(df[cols].isnull())*100
            print('{} - {}%'.format(cols, round(pct_missing,2)))
            
        
        
        print(df.columns)
        
        data_before = df.shape[0]
        df.dropna(axis=0, inplace = True)
        df.reset_index(drop=True, inplace =True)
        
        dropped = data_before - df.shape[0]
        print("dropped: "+str(dropped) + ", "+str(round((dropped/data_before)*100,2))+"%")
        print("total: "+str(df.shape[0]))        
        dates = []
        rel_countries = []
        for i in df['released']:
            # date = i.split("(")[0]
            date, rel_country = i.split("(")
            rel_country = rel_country.split(")")[0]
              
            dates.append(date)
            rel_countries.append(rel_country)
            
        print(df.isna().sum())
            
        df['released_date'] = dates
        df['released_date'] = pd.to_datetime(df['released_date'])
        df['month'] = df['released_date'].dt.month       
        df.drop(['released_date'], axis=1, inplace=True)       
        df['rel_country'] = rel_countries
        
        df['income']= df['gross'] - df['budget']
        # df.reset_index()
        df.drop_duplicates(inplace=True)
        
        df.gross = df.gross.astype(int)
        df.budget = df.budget.astype(int)
        df.votes = df.votes.astype(int)
        
        ratio = df['gross']/df['budget']
        df['ratio'] = ratio
        df.ratio = df.ratio.astype(int)
        
        df.reset_index(drop=True, inplace =True)
        print(df.describe().T)
        print(df.shape)
        print(df['genre'].unique())
        return df

    def checkGauss(self, df, variable_name):

        print("KS Test for {}: ".format(variable_name))
        x = df[str(variable_name)]
        ks_test = stats.kstest(x, 'norm')
        print(ks_test)
        # sns.displot(df, x=str(variable_name),bins=100)

    
    def plotDist(self, df, variable_name):
        fig = plt.figure(figsize=(25,10))        
        sns.displot(df[str(variable_name)],bins=12)

        plt.title('Distribución de la variable {}'.format(variable_name), loc='left', fontsize=18, pad=20)
        plt.show()        
    def preprocess(self):
        df = self.df.copy()
        variables_to_inspect = ['budget','gross','score','runtime', 'month','year']
        for variable_name in variables_to_inspect:
            self.checkGauss(df, variable_name)
            self.plotDist(df, variable_name)
            
        self.checkNumericCorrelation()
        self.checkAllCorrelation()
        # self.showPairPlot() 
        # !!!
        
    def showPairPlot(self):
        df = self.df.copy()
        sns.pairplot(df,diag_kind='hist')
        plt.show()
        
    def checkNumericCorrelation(self):
        df = self.df.copy()
        sns.heatmap(df.corr(method='pearson'), annot=True, vmin=-1, cmap='Blues')
        plt.title('Matriz de correlación (características numéricas)', loc='left', fontsize=18, pad=20)
        plt.show()
        
        

            
    def checkAllCorrelation(self):
        
        df = self.df.copy()

        for col in df.columns:
            if (df[col].dtype == 'object'):
                df[col] = df[col].astype('category')
                df[col] = df[col].cat.codes
        
        df.head()
        df_corr = df.corr(method='pearson')
        sns.heatmap(df_corr, annot=True, vmin=-1, cmap='YlOrBr')
        plt.title('Matriz de correlación completa', loc='left', fontsize=18, pad=20)
        plt.show()
        corr_pairs = df_corr.unstack()
        corr_pairs = corr_pairs[corr_pairs<1.0]
        high_corr = corr_pairs[corr_pairs>0.5]               
        top_corr = high_corr.sort_values(ascending=False) 
        print(top_corr.head(10))
    
#%%
#EDA:
    
    def plotGreaterIMDBvsGross(self):
        df = self.df.copy()
        best_score = df.sort_values('score',ascending=False).head(7)
        sns.barplot(x='name', y ='gross',hue='score', data=best_score)
   
        plt.xticks(rotation=90)    
        plt.xlabel('Película')
        plt.ylabel('Taquilla')
        plt.title('Taquilla de las películas mejor puntuadas', loc='left', fontsize=18, pad=20)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()

    def plotGreaterGrossvsIMDB(self):
        df = self.df.copy()
        best_gross = df.sort_values('gross',ascending=False).head(7)
        sns.barplot(x='name', y ='gross',hue='score', data=best_gross)
       
        plt.xticks(rotation=90) 

        plt.xlabel('Película')
        plt.ylabel('Taquilla')
        plt.title('Puntuación de las películas más taquilleras', loc='left', fontsize=18, pad=20)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()

    def plotCompaniesGrossandBudget(self):
        df = self.df.copy()
        companies = df.groupby('company')['gross','budget'].sum().reset_index().sort_values('gross',ascending=False).head(10)
        companies.plot(x='company',y=['budget','gross'],kind='bar',figsize=(12,5))
        plt.title("Taquilla y presupuesto por compañía",size=30)
        plt.xlabel("Compañía",size=15)
        plt.ylabel("Cantidad",size=15)
        plt.show()
        



        
        
    def plotCompaniesGrossandGenre(self):
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
        df = self.df.copy()
        # df_rating = df.groupby('rating','genre')['gross'].count().sort_values(ascending=False).head(5)
        sns.countplot(x = 'rating',data = df , hue='genre')
        plt.xlabel("Clasificación (edad)", fontsize=20, labelpad=10)
        # plt.ylabel("Taquilla", fontsize=20, labelpad=10)
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(10^3,10^3))
        # plt.xticks(rotation=90)
        plt.xticks(fontsize=16,rotation=90)
        plt.yticks(fontsize=16)
        plt.title('Cantidad de películas por clasificación y género', pad=20, fontsize=26)        
        plt.legend(loc='upper center')
        plt.show()


    def plotMoviesGrossandBudget(self):
        df = self.df.copy()
        movies = df.groupby('name')['gross','budget'].sum().reset_index().sort_values('gross',ascending=False).head(10)
        movies.plot(x='name',y=['budget','gross'],kind='bar',figsize=(12,5))
        plt.title("Taquilla y presupuesto por película",size=30)
        plt.xlabel("Película",size=15)
        plt.ylabel("Cantidad",size=15)
        plt.show()
    def plotNumbervsGenre(self):
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

        # fig = plt.figure(figsize=(12,8))
        df = self.df.copy()
        plt.scatter(x=df['budget'],y=df['gross'], c=df['score'], cmap="plasma_r")
        plt.xlabel('Presupuesto', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Taquilla vs Presupuesto', size=20, pad=10)
        cbar = plt.colorbar()
        cbar.set_label('IMDb', labelpad=10, size=14)
        plt.show()

    def processGenre(self):
        df = self.df.copy()
        movie_genre_gross = df.groupby(['genre'])['gross'].median().to_frame().reset_index().sort_values('gross', ascending = False).head(19)
        movie_genre_gross['genre'].to_list()

        # list_movie_genre_gross = sorted(movie_genre_gross['genre'].to_list())
        # list_movie_genre_gross
        print(movie_genre_gross)
        
        df = self.df.copy()
        movie_genre_IMDB = df.groupby(['genre'])['score'].median().to_frame().reset_index().sort_values('score', ascending = False).head(19)
        movie_genre_IMDB['genre'].to_list()
        # movie_genre_IMDB = sorted(movie_genre_IMDB['genre'].to_list())
        print(movie_genre_IMDB)
                

    def plotGenrevsGross(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(25,10))
        genre_vs_gross = df.groupby(['genre'])['gross'].median().reset_index().sort_values('gross', ascending = False).head(8)
        print(genre_vs_gross.head(8))
        sns.barplot(x='genre', y ='gross', data=genre_vs_gross)
        # plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Taquilla')
        plt.title('Taquilla media por género', loc='left', fontsize=18, pad=20)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()
        
    def plotRatings(self):
        df = self.df.copy()
        # dataset['rating'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(10,8))
        # plt.title('Rating percentages', fontsize = 20)
        # plt.tight_layout()
        # plt.show()
        
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
        df = self.df.copy()
        fig = plt.figure(figsize=(25,10))
        rating_df = df.sort_values('gross',ascending=False)
        ax_1 = sns.boxplot(x='rating', y='gross', data=rating_df, showfliers=False)
        # ax_1.tick_params(axis='x', rotation=-45)
        ax_1.tick_params(axis='both', labelsize=16)
        
        ax_1.set_title('Taquilla según clasificación', fontsize=24, pad=10)
        ax_1.set_ylabel('Taquilla', fontsize=20)
        ax_1.set_xlabel('Clasificación (edad)', fontsize=20)  
        plt.show()
        
    def plotRatingvsRatio(self): 
        df = self.df.copy()
        fig = plt.figure(figsize=(25,10))
        rating_df = df.sort_values('ratio',ascending=False)
        ax_1 = sns.boxplot(x='rating', y='ratio', data=rating_df, showfliers=False)
        # ax_1.tick_params(axis='x', rotation=-45)
        ax_1.tick_params(axis='both', labelsize=16)
        
        ax_1.set_title('Ratio Taquilla/Presupuesto según clasificación', fontsize=24, pad=10)
        ax_1.set_ylabel('Ratio Taquilla/Presupuesto', fontsize=20)
        ax_1.set_xlabel('Clasificación (edad)', fontsize=20)     
        plt.show()
        
    def plotGenrevsBudget(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(25,10))
        # ax = sns.barplot(self.df, x="genre", y="budget", hue='genre')
        # ax.set_title('Budget of different types of movie', fontsize=24, pad=10)
        # ax.set_ylabel('Budget', fontsize=20, labelpad=10)
        # ax.set_xlabel('Genre', fontsize=20, labelpad=10)
        genre_vs_budget = df.groupby(['genre'])['budget'].median().reset_index().sort_values('budget', ascending = False).head(8)
        print(genre_vs_budget.head(8))
        sns.barplot(x='genre', y ='budget', data=genre_vs_budget)
        # plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Presupuesto')
        plt.title('Presupuesto medio según género', loc='left', fontsize=18, pad=20)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))        
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()
        
    def plotGenrevsGrossBudgetRatio(self):
        
        fig = plt.figure(figsize=(25,10))
        df = self.df.copy()
        # ratio_df = df.assign(ratio = df['gross']/df['budget']).reset_index()
        mean_ratio_df = df.groupby(['genre'])['ratio'].median().to_frame().reset_index().sort_values('ratio',ascending=False).head(10)

        sns.barplot(x='genre', y ='ratio', data=mean_ratio_df)
        # plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Ratio Taquilla/Presupuesto')
        plt.title('Ratio Taquilla/Presupuesto según género', loc='left', fontsize=18, pad=20)
        plt.show()
        
    def plotGenrevsRatioBox(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(25,10))
        # ratio_df = df.assign(ratio = df['gross']/df['budget']).reset_index().sort_values('ratio',ascending=False)
        ratio_df = df.sort_values('ratio',ascending=False)
        ax_1 = sns.boxplot(x='genre', y='ratio', data=ratio_df, showfliers=False)
        # ax_1.tick_params(axis='x', rotation=-45)
        ax_1.tick_params(axis='both', labelsize=16)
        
        ax_1.set_title('Ratio Taquilla/Presupuesto según género', fontsize=24, pad=10)
        ax_1.set_ylabel('Ratio Taquilla/Presupuesto', fontsize=20)
        ax_1.set_xlabel('Género', fontsize=20)   
        
    def plotGenrevsYears(self):
        
        df = self.df.copy()
        # movie_genre_gross_by_year = df.groupby(['genre'])['gross'].mean().to_frame().reset_index().sort_values('year', ascending = False).head(19)
        # movie_genre_gross_by_year = df.groupby(['year','genre'])['gross'].mean().unstack().fillna(0)
        movie_genre_gross_by_year = df.groupby(['year','genre'])['gross'].median().unstack().fillna(0)
        # movie_genre_gross_by_year['genre'].to_list()
        # print(movie_genre_gross_by_year)
        
        fig = plt.figure(figsize=(25,10))
        ax_5 = sns.lineplot(data=movie_genre_gross_by_year, dashes=False, lw=3, palette='Set2')
        leg = ax_5.legend(fontsize=18)
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax_5.tick_params(axis='both', labelsize=16)
        ax_5.set_title('Tendencia en taquilla según género', fontsize=24, pad=10)
        ax_5.set_ylabel('Taquilla', fontsize=20, labelpad=10)
        ax_5.set_xlabel('Año', fontsize=20, labelpad=10)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))

    def plotRatingvsYears(self):
        
        df = self.df.copy()
        # movie_genre_gross_by_year = df.groupby(['genre'])['gross'].mean().to_frame().reset_index().sort_values('year', ascending = False).head(19)
        # movie_genre_gross_by_year = df.groupby(['year','genre'])['gross'].sum().unstack().fillna(0)
        movie_genre_gross_by_year = df.groupby(['year','rating'])['gross'].median().unstack().fillna(0)
        # movie_genre_gross_by_year['genre'].to_list()
        # print(movie_genre_gross_by_year)
        
        fig = plt.figure(figsize=(25,10))
        ax_5 = sns.lineplot(data=movie_genre_gross_by_year, dashes=False, lw=3, palette='Set2')
        leg = ax_5.legend(fontsize=18)
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax_5.tick_params(axis='both', labelsize=16)
        ax_5.set_title('Tendencia en taquilla según clasificación', fontsize=24, pad=10)
        ax_5.set_ylabel('Taquilla', fontsize=20, labelpad=10)
        ax_5.set_xlabel('Año', fontsize=20, labelpad=10)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        

    def plotGenrevsIMDB(self):
        fig = plt.figure(figsize=(25,10))
        ax = sns.histplot(self.df, x="genre", y="score",multiple='stack', bins=19,  hue='genre')
        ax.set_title('IMDb según género', fontsize=24, pad=10)
        ax.set_ylabel('IMDb', fontsize=20, labelpad=10)
        ax.set_xlabel('Género', fontsize=20, labelpad=10)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()
        
    def plotIMDBvsYear(self):
        df = self.df.copy()
        imdb_by_year_and_genre = df.groupby(['year','genre'])['score'].median().unstack().fillna(method="ffill")
         
        fig = plt.figure(figsize=(25,10))
        ax_5 = sns.lineplot(data=imdb_by_year_and_genre, dashes=False, lw=3, palette='Set2')
        leg = ax_5.legend(fontsize=18)
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax_5.tick_params(axis='both', labelsize=16)
        ax_5.set_title('Tendencia IMDb según género', fontsize=24, pad=10)
        ax_5.set_ylabel('IMDB', fontsize=20, labelpad=10)
        ax_5.set_xlabel('Año', fontsize=20, labelpad=10)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))   
        
    def plotIMDBvsGenre(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(25,10))
        ax_1 = sns.boxplot(x='genre', y='score', data=df, showfliers=False)
        # ax_1.tick_params(axis='x', rotation=-45)
        ax_1.tick_params(axis='both', labelsize=16)
        
        ax_1.set_title('IMDb según género', fontsize=24, pad=10)
        ax_1.set_ylabel('IMDb', fontsize=20)
        ax_1.set_xlabel('Género', fontsize=20)

    def plotGrossvsGenreBox(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(25,10))
        ax_1 = sns.boxplot(x='genre', y='gross', data=df, showfliers=False)
        # ax_1.tick_params(axis='x', rotation=-45)
        ax_1.tick_params(axis='both', labelsize=16)
        
        ax_1.set_title('Taquilla según género', fontsize=24, pad=10)
        ax_1.set_ylabel('Taquilla', fontsize=20)
        ax_1.set_xlabel('Género', fontsize=20) 

    def plotBudgetvsGenreBox(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(25,10))
        ax_1 = sns.boxplot(x='genre', y='budget', data=df, showfliers=False)
        # ax_1.tick_params(axis='x', rotation=-45)
        ax_1.tick_params(axis='both', labelsize=16)
        
        ax_1.set_title('Presupuesto medio según género', fontsize=24, pad=10)
        ax_1.set_ylabel('Presupuesto', fontsize=20)
        ax_1.set_xlabel('Género', fontsize=20)         
        
    def getIncomesRanking(self):
        df = self.df.copy()
        movie_incomes_by_genre = df.groupby(['genre'])['income'].mean().to_frame().reset_index().sort_values('income', ascending = False).head(19)
        movie_incomes_by_genre['genre'].to_list()
        print(movie_incomes_by_genre)
        # self.plotIncomes()
    
        
    def plotIncomes(self):
        df = self.df.copy()
        movie_genre_gross_by_year = df.groupby(['year','genre'])['income'].mean().unstack().fillna(0)
        # movie_genre_gross_by_year['genre'].to_list()
        # print(movie_genre_gross_by_year)
        
        fig = plt.figure(figsize=(25,10))
        ax_5 = sns.lineplot(data=movie_genre_gross_by_year, dashes=False, lw=3, palette='Set2')
        leg = ax_5.legend(fontsize=18)
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax_5.tick_params(axis='both', labelsize=16)
        ax_5.set_title('Tendencia de beneficio neto según el género', fontsize=24, pad=10)
        ax_5.set_ylabel('Beneficio', fontsize=20, labelpad=10)
        ax_5.set_xlabel('Año', fontsize=20, labelpad=10)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()
        

    def getBestMonthByGenre(self):
        df = self.df.copy()
        movie_gross_by_month = df.groupby(['month','genre'])['gross'].median().to_frame().reset_index().sort_values('gross', ascending = False).head(19)
        movie_gross_by_month['genre'].to_list()
        print(movie_gross_by_month)
        # self.plotBestMothbyGenre()

    def plotBestMothbyGenre(self):
        df = self.df.copy()
        movie_gross_by_month = df.groupby(['month','genre'])['gross'].median().unstack().fillna(0)
        fig = plt.figure(figsize=(10,8))
        ax_5 = sns.heatmap(movie_gross_by_month, cbar_kws={'label': 'Taquilla'})
        ax_5.set_title('Taquilla según género y mes de estreno', fontsize=20, pad=10)
        ax_5.set_xlabel('Género', fontsize=16, labelpad=10)
        ax_5.set_ylabel('Mes', fontsize=16, labelpad=10)
        ax_5.tick_params(axis='both', labelsize=10)
        plt.show()
    def plotBestMothbyYear(self):
        df = self.df.copy()
        movie_gross_by_month = df.groupby(['month','year'])['gross'].median().unstack().fillna(0)
        fig = plt.figure(figsize=(10,8))
        ax_5 = sns.heatmap(movie_gross_by_month, cbar_kws={'label': 'Taquilla'})
        ax_5.set_title('Taquilla según año y mes de estreno', fontsize=20, pad=10)
        ax_5.set_xlabel('Año', fontsize=16, labelpad=10)
        ax_5.set_ylabel('Mes', fontsize=16, labelpad=10)
        ax_5.tick_params(axis='both', labelsize=10)
        plt.show()   
        
    def plotTop10GrossMovies(self):
        df = self.df.copy()
        df = df.sort_values(by=['gross'],ascending=False).head(10)
        sns.barplot(x='name', y ='gross', data=df)
        plt.xticks(rotation=90)
        plt.xlabel('Película')
        plt.ylabel('Taquilla')
        plt.title('Top 10 de películas más taquilleras', loc='left', fontsize=18, pad=20)
        plt.show()

    def plotRemakevsGross(self):
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
        

    def plotProductionCountryvsReleaseContry(self):
        df = self.df.copy()
        movie_gross_by_country = df.groupby(['rel_country','country'])['gross'].median().unstack().fillna(0)
        fig = plt.figure(figsize=(10,8))
        ax_5 = sns.heatmap(movie_gross_by_country, cbar_kws={'label': 'Taquilla'})
        ax_5.set_title('Taquilla según el país de estreno y el país de producción', fontsize=20, pad=10)
        ax_5.set_xlabel('País de estreno', fontsize=16, labelpad=10)
        ax_5.set_ylabel('País de producción', fontsize=16, labelpad=10)
        ax_5.tick_params(axis='both', labelsize=10)
        plt.show() 
        
        
    def plotGrossvsIMDB(self):
        df = self.df.copy()
        plt.scatter(y=df['gross'],x=df['score'])
        plt.ylabel('Taquilla', labelpad=10, size=14)
        plt.xlabel('IMDb', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Taquilla según IMDb', size=20, pad=10)
        # cbar = plt.colorbar()
        # cbar.set_label('IMDb score', labelpad=10, size=14)
        plt.show()


    def plotBudgetvsIMDB(self):
        df = self.df.copy()
        plt.scatter(y=df['budget'],x=df['score'])
        plt.ylabel('Presupuesto', labelpad=10, size=14)
        plt.xlabel('IMDb', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Presupuesto según IMDb', size=20, pad=10)
        # cbar = plt.colorbar()
        # cbar.set_label('IMDb score', labelpad=10, size=14)
        plt.show()
        
    def plotNumbervsYear(self):
        df = self.df.copy()
        movies_per_year = df.groupby(['year']).count()
        fig = plt.figure(figsize=(10,8))
        sns.lineplot(x=movies_per_year.index, y='name', data=movies_per_year, linewidth=4)
        plt.xticks(rotation=45)
        plt.xlabel('Año')
        plt.ylabel('Películas')
        plt.title('Cantidad de películas estrenadas por año', loc='left', fontsize=18, pad=20)
        plt.show()
        # print(movies_per_year)
        
    def plotCountryvsGross(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(10,8))
        country_mean_gross = df.groupby(['country'])['gross'].median().to_frame().reset_index().sort_values('gross', ascending = False).head(8)
        sns.barplot(x='country', y ='gross', data=country_mean_gross)
        plt.xticks(rotation=45)
        plt.xlabel('País de producción')
        plt.ylabel('Taquilla')
        plt.title('Taquilla media según el país de producción', loc='left', fontsize=18, pad=20)
        plt.show()
                
    def plotProductionCountryvsBudget(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(10,8))
        country_mean_gross = df.groupby(['country'])['budget'].median().to_frame().reset_index().sort_values('budget', ascending = False).head(8)
        sns.barplot(x='country', y ='budget', data=country_mean_gross)
        plt.xticks(rotation=45)
        plt.xlabel('País de producción')
        plt.ylabel('Presupuesto')
        plt.title('Presupuesto medio según el país de producción', loc='left', fontsize=18, pad=20)
        plt.show()
                
    def plotReleaseCountryvsGross(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(10,8))
        country_mean_gross = df.groupby(['rel_country'])['gross'].median().to_frame().reset_index().sort_values('gross', ascending = False).head(8)
        sns.barplot(x='rel_country', y ='gross', data=country_mean_gross)
        plt.xticks(rotation=45)
        plt.xlabel('País de estreno')
        plt.ylabel('Taquilla')
        plt.title('Taquilla media según el país de estreno', loc='left', fontsize=18, pad=20)
        plt.show()        
        
    def plotCountryvsNumber(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(10,8))
        country_production = df.groupby(['country']).count().sort_values('name', ascending = False).head(8)
        sns.barplot(x=country_production.index, y ='name', data=country_production)
        plt.xticks(rotation=45)
        plt.xlabel('País de producción')
        plt.ylabel('Películas')
        plt.title('Cantidad de películas producidas por país', loc='left', fontsize=18, pad=20)
        plt.show()



    def plotCountryvsNumberByYear(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(10,8))

        country_production = df.groupby(['country','year']).count().groupby(['country'])['name'].median().to_frame().sort_values('name', ascending = False).head(8)
        print(country_production.head(8))        
        sns.barplot(x=country_production.index, y = 'name', data=country_production)

        plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Películas')
        plt.title('Cantidad media de películas producidas al año por país', loc='left', fontsize=18, pad=20)
        plt.show() 




    def plotGenrevsNumber(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(10,8))
        country_production = df.groupby(['genre']).count().sort_values('name', ascending = False).head(8)
        sns.barplot(x=country_production.index, y ='name', data=country_production)
        plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Películas')
        plt.title('Cantidad de películas por género', loc='left', fontsize=18, pad=20)
        plt.show()                



    def plotGenrevsNumberByYear(self):
        df = self.df.copy()
        fig = plt.figure(figsize=(10,8))

        country_production = df.groupby(['genre','year']).count().groupby(['genre'])['name'].median().to_frame().sort_values('name', ascending = False).head(8)
        print(country_production.head(8))        
        sns.barplot(x=country_production.index, y = 'name', data=country_production)

        plt.xticks(rotation=45)
        plt.xlabel('Género')
        plt.ylabel('Películas')
        plt.title('Cantidad media de películas por género y año', loc='left', fontsize=18, pad=20)
        plt.show() 
        
        
    def plotRuntimevsGross(self):
        df = self.df.copy()
        plt.scatter(x=df['runtime'],y=df['gross'], c=df['score'], cmap="plasma_r")
        plt.xlabel('Duración (en min)', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Taquilla según duración', size=20, pad=10)
        cbar = plt.colorbar()
        cbar.set_label('IMDb', labelpad=10, size=14)
        plt.show()

    def plotRuntimevsBudget(self):
        df = self.df.copy()
        plt.scatter(x=df['runtime'],y=df['budget'], c=df['score'], cmap="plasma_r")
        plt.xlabel('Duración (en min)', labelpad=10, size=14)
        plt.ylabel('Presupuesto', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Presupuesto según duración', size=20, pad=10)
        cbar = plt.colorbar()
        cbar.set_label('IMDb', labelpad=10, size=14)
        plt.show()        
 
        
 
    def plotYearlyGrossPercentage(self):
        df = self.df.copy()
        year_rev = df.groupby('year')['gross'].sum().pct_change().to_frame().reset_index().sort_values('year', ascending = True)
        year_rev_pct = year_rev.assign(year_pct = year_rev['gross']*100).reset_index()
        print(year_rev_pct.head(8))
        sns.barplot(x='year', y ='year_pct', data=year_rev_pct)
        plt.xticks(rotation=90)
        plt.xlabel('Año')
        plt.ylabel('Taquilla (%)')
        plt.title('Taquilla (%) por año', loc='left', fontsize=18, pad=20)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(10^3,10^3))
        plt.show()
        # year_rev_pct = year_rev.pct_change()

        # rev_df = pd.DataFrame({
        #     'Yearly Revenue': year_rev, 
        #     'Yearly Revenue (%)': year_rev_pct * 100
        #     })
        

     
        
#%%
# CORRELATION ANALYSIS

        
    def plotRuntimevsGrossRegression(self):
        df = self.df.copy()
        sns.regplot(x='runtime',y='gross',data=df,scatter_kws={"color":"blue"},line_kws={"color":"red"})
        plt.xlabel('Duración (en min)', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Taquilla según duración', size=20, pad=10)
        # cbar = plt.colorbar()
        # cbar.set_label('IMDb score', labelpad=10, size=14)
        plt.show()

        
    def plotRuntimevsBudgetRegression(self):
        df = self.df.copy()
        sns.regplot(x='runtime',y='budget',data=df,scatter_kws={"color":"red"},line_kws={"color":"green"})
        plt.xlabel('Duración (en min)', labelpad=10, size=14)
        plt.ylabel('Presupuesto', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Presupuesto según duración', size=20, pad=10)
        # cbar = plt.colorbar()
        # cbar.set_label('IMDb score', labelpad=10, size=14)
        plt.show()

    
    def plotBudgetvsGrossRegression(self):

        # fig = plt.figure(figsize=(12,8))
        df = self.df.copy()
        # plt.scatter(x=df['budget'],y=df['gross'], c=df['score'], cmap="plasma_r")
        sns.regplot(x='budget',y='gross',data=df,scatter_kws={"color":"blue"},line_kws={"color":"red"})
        plt.xlabel('Presupuesto', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Taquilla según presupuesto', size=20, pad=10)
        # cbar = plt.colorbar()
        # cbar.set_label('IMDb score', labelpad=10, size=14)
        plt.show()
        

    def plotVotesvsGrossRegression(self):

        # fig = plt.figure(figsize=(12,8))
        df = self.df.copy()
        # plt.scatter(x=df['budget'],y=df['gross'], c=df['score'], cmap="plasma_r")
        sns.regplot(x='votes',y='gross',data=df,scatter_kws={"color":"blue"},line_kws={"color":"red"})
        plt.xlabel('Votos', labelpad=10, size=14)
        plt.ylabel('Taquilla', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('Taquilla según votos', size=20, pad=10)
        # cbar = plt.colorbar()
        # cbar.set_label('IMDb score', labelpad=10, size=14)
        plt.show()

        
    def plotIMDBvsGrossRegression(self):
        df = self.df.copy()
        sns.regplot(x='gross',y='score',data=df,scatter_kws={"color":"blue"},line_kws={"color":"red"})
        plt.xlabel('Taquilla', labelpad=10, size=14)
        plt.ylabel('IMDb', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('IMDb según taquilla', size=20, pad=10)
        # cbar = plt.colorbar()
        # cbar.set_label('IMDb score', labelpad=10, size=14)
        plt.show()

    def plotIMDBvsBudgetRegression(self):
        df = self.df.copy()
        sns.regplot(x='budget',y='score',data=df,scatter_kws={"color":"red"},line_kws={"color":"green"})
        plt.xlabel('Presupuesto', labelpad=10, size=14)
        plt.ylabel('IMDb', labelpad=10, size=14)
        # plt.ticklabel_format(axis="both", style="sci", scilimits=(10^3,10^3))
        plt.title('IMDb según presupuesto', size=20, pad=10)
        # cbar = plt.colorbar()
        # cbar.set_label('IMDb score', labelpad=10, size=14)
        plt.show()
        


#%%
#Read & clean data:
movieExplorer = MovieExplorer()

#%%
#Explore dataset:
# By years (2020==pandemic):
movieExplorer.plotRatingvsYears()
movieExplorer.plotNumbervsYear()
movieExplorer.plotYearlyGrossPercentage()
movieExplorer.plotCountryvsNumberByYear()
#By year & month:
movieExplorer.plotBestMothbyYear()    
#By film:
movieExplorer.plotTop10GrossMovies()
movieExplorer.plotMoviesGrossandBudget()
movieExplorer.plotRemakevsGross()
#By companies:
movieExplorer.plotCompaniesGrossandBudget()
movieExplorer.plotCompaniesGrossandGenre()
movieExplorer.plotCompaniesGrossandGenreMean()

#By rating:
movieExplorer.plotRatings()  
movieExplorer.plotRatingByGenere()
movieExplorer.plotRatingvsGross()  
movieExplorer.plotRatingvsRatio()

#By genre:
movieExplorer.plotNumbervsGenre()
movieExplorer.processGenre()
movieExplorer.plotGenrevsNumber()
movieExplorer.plotGenrevsGross()
movieExplorer.plotGenrevsBudget()
movieExplorer.plotGenrevsGrossBudgetRatio()

movieExplorer.plotGrossvsGenreBox()
movieExplorer.plotBudgetvsGenreBox()
movieExplorer.plotGenrevsRatioBox()

movieExplorer.getBestMonthByGenre()
movieExplorer.plotBestMothbyGenre()

movieExplorer.plotGenrevsYears()
movieExplorer.plotGenrevsNumberByYear()
movieExplorer.plotGenrevsIMDB()

#By incomes (gross-budget):
movieExplorer.getIncomesRanking()
movieExplorer.plotIncomes()

#By score:
movieExplorer.plotGreaterIMDBvsGross()
movieExplorer.plotGreaterGrossvsIMDB()
movieExplorer.plotBudgetvsGrossvsScore()
movieExplorer.plotGrossvsIMDB()
movieExplorer.plotBudgetvsIMDB()
movieExplorer.plotIMDBvsYear()
movieExplorer.plotIMDBvsGenre()

#By country:
movieExplorer.plotProductionCountryvsBudget()
movieExplorer.plotCountryvsGross()
movieExplorer.plotCountryvsNumber()
movieExplorer.plotReleaseCountryvsGross()
movieExplorer.plotProductionCountryvsReleaseContry()

#By runtime:
movieExplorer.plotRuntimevsGross()
movieExplorer.plotRuntimevsBudget()

#%%
#Regression models:
movieExplorer.plotRuntimevsBudgetRegression()
movieExplorer.plotRuntimevsGrossRegression()
movieExplorer.plotBudgetvsGrossRegression()
movieExplorer.plotIMDBvsGrossRegression()
movieExplorer.plotIMDBvsBudgetRegression()
movieExplorer.plotVotesvsGrossRegression()