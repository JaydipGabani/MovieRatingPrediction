# RatingPredictionSystem
Utilizing the Open Movie Database API and Netflix Data for predicting rating of a movie.

Instruction to run:
1. Run the createVectorizedData.py. This will generate two txt files- vectorizedModelInput.txt, vectorizedRatings.txt. The program will also generate vectorData.csv to observe the changes in csv format. Make sure to make editings in below line of the code. The value you provide here will reduce the size of the data to that extent. 

  dataframeMovies = dataframeMovies.drop(dataframeMovies.sample(frac=.95).index)

  The above line is intended to run the vectorization (a time consuming process) on low volume of data so as to decrease the runtime.

  Once the model is finalized, please comment out this line to to convert all the data into vectorized form.

2. Run the finalModel.py to get the accuracy of the model. The file will provide the accuracy in console and will also create a figure named - TestAccuracyforgivenerrorrange.png.


Below files have already been run, and need not to be reran. But if you want to explore on how it's done, then please feel free to run them again. These will take longer to run:

1. The data has already been fetched from OMDB API in movie_details.csv by the code in dataFetching.ipynb. And hence, it is not needed to be fetched again.

2. The base models (used as estimators) in finalModel.py are already set to a certain hyperparameter values. This values are obtained from tuning.py and can be modified upon analysis of the output obtained by running tuning.py



