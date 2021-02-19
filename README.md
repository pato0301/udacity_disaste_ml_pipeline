# Disaster Response Pipeline Project

Figure Eight has provided data related to messages, categorized into different classifications, that have been received during emergencies/disasters. This project try to recognize these categories in order to cater for quicker responses to the emergency messages. Using machine learning techniques, (Random Forest Classifier) we shold be able to predict the category.

### The process was carried out as follows:

1. Data Processing Assessing and cleaning the data, so that it can be utilized by machine learning algorithms. See details in the ETL Notebook.

2. Model training Data was passed through a pipeline and a prediction model is made. See details in the ML Notebook.

3. Prediction and Visualization Making a web app for prediction and visualization, where user may try some emergency messages and see visualization of distribution of genres and categories.

# Project motivation

For this project, I use the Figure Eight disaster response data to build a classifier that flags messages for various emergency services.

# Install

This project requires Python 3.x and the following Python libraries installed:

- NumPy
- Pandas
- Nltk
- Flask
- Sklearn
- Sqlalchemy
- Sys
- Re
- Pickle
- json
- plotly

# Instructions to run the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# File descriptions

- ETL Pipeline Preparation.ipynb - data parsing, preparation, and saving
- ML Pipeline Preparation.ipynb - classification (main file). DOES requrie Google's pretrained word2Vec to run fully (but you can always just unpickle).
- ./pickles/pipeline_advanced3.pkl - pre-trained classifier that can be unpickled
- ./pickles/words_mappings.pkl - cached words mappings
- ./raw_data/categories.csv - message categories data
- ./raw_data/messages.csv - messages themselves
- ./web/process_data.py - file for command-line ETL
- ./web/train_classifier.py - file for command-line classifier training. Does NOT require Google's pre-trained word2Vec (uses cached mappings)
- ./web/words_mappings.pkl - cached mappings for train_classifier.py
- ./web/run.py - file to run with flask (used in conjunction with Udacity's IDE, won't run by itself)
- ./py_scripts/my_etl_pipeline/etlfuncs.py - my ETL helper functions
- ./py_scripts/my_etl_pipeline/etl_pipeline.py - another implementation of the ETL pipeline

# Licensing, Authors, Acknowledgements

You are free to use the code as you like.
