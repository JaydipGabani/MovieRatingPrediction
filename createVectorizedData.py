import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import scale

dataframeMovies = pd.read_csv('movie_details.csv')
dataframeMovies = dataframeMovies[['NeflixMovieTitle','Genre','Language','Country','Rated', 'Production', 'Runtime', 'Type','Director','Actors','Writer','IMDBRating']]
#dataframeMovies = dataframeMovies[['NeflixMovieTitle','Genre','Language','Country','Rated', 'Production', 'Runtime', 'Type','Actors','Writer','IMDBRating']]
dataframeMovies = dataframeMovies.dropna(subset=['NeflixMovieTitle','Genre','Language','Country','Rated', 'Production', 'Runtime', 'Type','Director','Actors','Writer','IMDBRating'])
#dataframeMovies = dataframeMovies.dropna(subset=['NeflixMovieTitle','Genre','Language','Country','Rated', 'Production', 'Runtime', 'Type','Actors','Writer','IMDBRating'])
dataframeMovies = dataframeMovies[dataframeMovies.Type == "movie"]
# dataframeMovies = dataframeMovies[dataframeMovies.Language == "English"]
# dataframeMovies = dataframeMovies[dataframeMovies.Country == "USA"]
# For testing the code, uncomment below line to reduce the size of input data
dataframeMovies = dataframeMovies.drop(dataframeMovies.sample(frac=.95).index)

dataframeMovies.reset_index(drop = True, inplace = True)
dataframeMovies = dataframeMovies.drop_duplicates(subset='NeflixMovieTitle', keep="first")
dataframeMovies = dataframeMovies[['Genre','Language','Country','Rated', 'Production', 'Runtime','Director','Actors','Writer','IMDBRating']]
#dataframeMovies = dataframeMovies[['Genre','Language','Country','Rated', 'Production', 'Runtime','Actors','Writer','IMDBRating']]
dataframeMovies.reset_index(drop = True, inplace = True)
labels = dataframeMovies['IMDBRating']
categoricalCols = ['Genre','Language','Country','Rated','Production','Actors']
print('Total number of input rows after removing duplicates and null vales from all columns in the data = '+str(len(dataframeMovies)))

for i in range(len(dataframeMovies)):
    dataframeMovies['Writer'][i] = dataframeMovies['Writer'][i].split(',')
    dataframeMovies['Actors'][i] = dataframeMovies['Actors'][i].split(',')
    dataframeMovies['Genre'][i] = dataframeMovies['Genre'][i].split(',')
    dataframeMovies['Director'][i] = dataframeMovies['Director'][i].split(',')
    dataframeMovies['Language'][i] = dataframeMovies['Language'][i].split(',')[0]
    dataframeMovies['Country'][i] = dataframeMovies['Country'][i].split(',')[0]
    dataframeMovies['Runtime'][i] = int(dataframeMovies['Runtime'][i].split(" ")[0])
    for writer in range (len(dataframeMovies['Writer'][i])):
        dataframeMovies['Writer'][i][writer] = dataframeMovies['Writer'][i][writer].split('(')[0]
        dataframeMovies['Writer'][i][writer] = 'writer '+dataframeMovies['Writer'][i][writer].strip()
    for actor in range (len(dataframeMovies['Actors'][i])):
        dataframeMovies['Actors'][i][actor] = dataframeMovies['Actors'][i][actor].strip()
    for director in range (len(dataframeMovies['Director'][i])):
        dataframeMovies['Director'][i][director] = 'director '+dataframeMovies['Director'][i][director].strip()

# Create separate columns for each genre and set value as 0 or 1
print('Converting Genres into columns...')
dataframeGenresList = list(dataframeMovies['Genre'])
genres = list(dataframeGenresList[0])
for index in range(1,len(dataframeGenresList)):
    genres.extend(dataframeGenresList[index])
# create a set of all the unique possible value of genres
genres = list(sorted(set(genres)))
dfGenres = pd.DataFrame(index = dataframeMovies.index, columns = genres)
for i in dfGenres.index:
    currGen = dataframeMovies['Genre'][i]
    for col in dfGenres.columns:
        if col in currGen:
            dfGenres[col][i] = 1
        else:
            dfGenres[col][i] = 0

# Create separate columns for each actor and set value as 0 or 1
print('Converting Actors into columns...')
dataframeActorsList = list(dataframeMovies['Actors'])
actors = list(dataframeActorsList[0])
for index in range(1,len(dataframeActorsList)):
    actors.extend(dataframeActorsList[index])
# create a set of all the unique possible value of actors
actors = list(sorted(set(actors)))
dfActors = pd.DataFrame(index = dataframeMovies.index, columns = actors)
for i in dfActors.index:
    currAct = dataframeMovies['Actors'][i]
    for col in dfActors.columns:
        if col in currAct:
            dfActors[col][i] = 1
        else:
            dfActors[col][i] = 0

# Create separate columns for each writer and set value as 0 or 1
print('Converting Writers into columns...')
dataframeWritersList = list(dataframeMovies['Writer'])
writers = list(dataframeWritersList[0])
for writer in range(1,len(dataframeWritersList)):
    writers.extend(dataframeWritersList[writer])
# create a set of all the unique possible value of writers
writers = list(sorted(set(writers)))
dfWriters = pd.DataFrame(index = dataframeMovies.index, columns = writers)
for i in dfWriters.index:
    currWriter = dataframeMovies['Writer'][i]
    for col in dfWriters.columns:
        if col in currWriter:
            dfWriters[col][i] = 1
        else:
            dfWriters[col][i] = 0
        
# Create separate columns for each director and set value as 0 or 1
print('Converting Directors into columns...')
dataframeDirectorsList = list(dataframeMovies['Director'])
directors = list(dataframeDirectorsList[0])
for director in range(1,len(dataframeDirectorsList)):
    directors.extend(dataframeDirectorsList[director])
# create a set of all the unique possible value of directors
directors = list(sorted(set(directors)))
dfDirectors = pd.DataFrame(index = dataframeMovies.index, columns = directors)
for i in dfDirectors.index:
    currDirector = dataframeMovies['Director'][i]
    for col in dfDirectors.columns:
        if col in currDirector:
            dfDirectors[col][i] = 1
        else:
            dfDirectors[col][i] = 0

dataframeMovies = dataframeMovies.drop('Genre', axis=1)
dataframeMovies = dataframeMovies.drop('Actors', axis=1)
dataframeMovies = dataframeMovies.drop('Writer', axis=1)
dataframeMovies = dataframeMovies.drop('Director', axis=1)
dataframeMovies = dataframeMovies.join(dfGenres)
dataframeMovies = dataframeMovies.join(dfActors)
dataframeMovies = dataframeMovies.join(dfWriters)
dataframeMovies = dataframeMovies.join(dfDirectors)

print(dataframeMovies)
dataframeMovies.to_csv('vectorData.csv', index=False)

dictVectorizer = DictVectorizer()
vectorizeColumns = ['Language','Country','Rated','Production']
df = pd.DataFrame(dictVectorizer.fit_transform(dataframeMovies[vectorizeColumns].to_dict(orient='records')).toarray())
df.columns = dictVectorizer.get_feature_names()
df.index = dataframeMovies.index
dataframeMovies = dataframeMovies.drop(vectorizeColumns, axis=1)
dataframeMovies = dataframeMovies.join(df)

print('Scaling all the vectorized data')
dataframeIMDBRatings = dataframeMovies['IMDBRating']
dataframeMovies = dataframeMovies.drop('IMDBRating', axis=1)
dataframeMovies = dataframeMovies.join(dataframeIMDBRatings)
dataframeMovies.to_csv('vectorData.csv', index=False)
listAllColumns = list(dataframeMovies.columns)
vectorizeData = np.array(dataframeMovies)
vectorizeData = vectorizeData.astype(np.float)
vectorizeRatings = vectorizeData[:,-1]
vectorizeData = scale(vectorizeData)

print('Generating final output')
np.savetxt('vectorizedModelInput.txt',vectorizeData)
np.savetxt('vectorizedRatings.txt',vectorizeRatings)
