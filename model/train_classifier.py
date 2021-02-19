import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

def load_data(database_filepath):
    """
        Loads data from the Database 
        Into arrays of X (feature) and y (labels)
    """
    #connecting to Database and load data
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable', engine)
    
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    X = df['message'].values
    y = df.iloc[:,4:]
    category_names = y.columns
    y = y.values

    return X, y, category_names


def tokenize(text):
    """
        Tokenize the features using NLP.
    """
    
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Classifier is build using pipeline. GridSearch is used to fine tune the paramenter.
    Only two were fine tuned but more can be used. The function return the model to be trained
    """
    #using Pipeline to build classifier
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    #optimize classifier with GridSearch
    parameters = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
              'classifier__estimator__n_estimators': [10, 20, 40]}
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluates the trained model, printing precision, recall, f1_score and support
    """
    y_prediction_test = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[:,i], y_prediction_test[:,i]))


def save_model(model, model_filepath):
    """
        Saves the model as a pickle file so that it can be used as a classifier in production
    """
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()