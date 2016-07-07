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
    return np.array(user_movie), np.array(ratings)


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
