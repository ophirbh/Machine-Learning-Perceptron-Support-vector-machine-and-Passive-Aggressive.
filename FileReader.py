import numpy as np


def read_from_file(file_name):
    data_file = open(file_name, "r")
    data = []

    for line in data_file:
        line_sepetated = line.split("\n")
        line_sepetated = line_sepetated[0].split(",")

        params = []

        # Put the data in an array
        for word in line_sepetated:
            params.append(word)

        data.append(params)

    data_file.close()
    return data

# read_from_file("train_x.txt")