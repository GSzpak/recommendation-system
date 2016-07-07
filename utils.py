import csv

import click


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
