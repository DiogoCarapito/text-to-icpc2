# script to compare 2 predictions csv files and checko % ot match

import click
import pandas as pd


@click.command()
@click.option(
    "--runid1", default="74fa84bab5a148d2b37ba79ca3dfda64", help="file 1", required=True
)
@click.option(
    "--runid2", default="a7fc0297663d4ff79f00de27c4a07e34", help="file 2", required=True
)
def main(runid1, runid2):
    file_1 = f"correct_predictions/correct_predictions_{runid1}.csv"
    file_2 = f"correct_predictions/correct_predictions_{runid2}.csv"

    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    # check if the dataframes match code and get the % of match
    match = df1["code"].equals(df2["code"])
    print(f"Match: {match}")


if __name__ == "__main__":
    main()
