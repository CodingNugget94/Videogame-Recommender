import re

import numpy as np
import pandas as pd
import flask
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Dataset import
dataset = pd.read_csv('Dataset//Videogames Data.csv', encoding='mac_roman')


# Build model with matrix
def build_model():
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(dataset['combined_features'])
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(count_matrix)
    return dataset, model, count_matrix


def recommend_game(searched_game):
    data, model, count_matrix = build_model()
    if searched_game in dataset['Title'].values:
        choice_index = dataset[dataset['Title'] == searched_game].index.values[0]
        distances, indices = model.kneighbors(count_matrix[choice_index], n_neighbors=10)
        recommended_games = []
        for i in indices.flatten():
            recommended_games.append(data[dataset.index == i]['Title'].values[0].Title())
        return recommended_games
    elif dataset['Title'].str.contains(searched_game).any():

        # getting list of similar movie names as choice.
        similar_names = list(str(s) for s in dataset['Title'] if searched_game in str(s))
        # sorting the list to get the most matched movie name.
        similar_names.sort()
        # taking the first movie from the sorted similar movie name.
        new_choice = similar_names[0]
        print(new_choice)
        # getting index of the choice from the dataset
        choice_index = dataset[dataset['Title'] == new_choice].index.values[0]
        # getting distances and indices of 16 mostly related movies with the choice.
        distances, indices = model.kneighbors(count_matrix[choice_index], n_neighbors=10)
        # creating movie list
        recommended_games = []
        for i in indices.flatten():
            recommended_games.append(dataset[dataset.index == i]['Title'].values[0].Title())
        return recommended_games
        # If no name matches then this else statement will be executed.
    else:
        return "Game not found!"


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('game_search.html')


@app.route("/Search")
def search_movies():
    # getting user input
    searched_game = request.args.get('movie')
    # removing all the characters except alphabets and numbers.
    choice = re.sub("[^a-zA-Z1-9]", "", searched_game).lower()
    # passing the choice to the recommend() function
    games = recommend_game(choice)
    # if rocommendation is a string and not list then it is else part of the
    # recommend() function.
    if type(games) == type('string'):
        return render_template('game_suggestions.html', games=games, s='opps')
    else:
        return render_template('game_suggestions.html', games=games)


if __name__ == "__main__":
    app.run(debug=False)
