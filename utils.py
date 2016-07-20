from collections import defaultdict
import csv

import click
import numpy as np


def read_ratings_from_csv(input_file):
    user_movie = []
    ratings = []
    with open(input_file, "r") as input:
        reader = csv.reader(input)
        for row in reader:
            user_id, movie_id, rating, _timestamp = row
            user_movie.append([user_id, movie_id])
            ratings.append(rating)
    return np.array(user_movie), np.array(ratings).astype(np.int32)


def build_user_utility_matrix(input_file):
    user_ratings = defaultdict(dict)
    last_user_id = 0
    last_movie_id = 0
    with open(input_file, "r") as input:
        reader = csv.reader(input)
        for row in reader:
            user_id, movie_id, rating, _timestamp = row
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            last_user_id = max(last_user_id, user_id)
            last_movie_id = max(last_movie_id, movie_id)
            user_ratings[user_id][movie_id] = rating
    user_ratings_list = user_ratings.items()
    user_utility_matrix = np.zeros((last_user_id + 1, last_movie_id + 1), dtype=np.int32)
    for user_id, ratings in user_ratings_list:
        for movie_id, rating in ratings.iteritems():
            user_utility_matrix[user_id, movie_id] = rating
    return user_utility_matrix


def build_item_utility_matrix(input_file):
    return build_user_utility_matrix(input_file).transpose()


@click.command()
@click.argument('input-file', type=click.Path(exists=True))
@click.argument('output-file', type=click.Path(exists=False))
def convert_to_csv(input_file, output_file):
    with open(input_file, "r") as input, open(output_file, "w") as output:
        writer = csv.writer(output)
        for row in input:
            row = row.splitlines()[0]
            row = row.split("::")
            writer.writerow(row)


if __name__ == '__main__':
    convert_to_csv()
