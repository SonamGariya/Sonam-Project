
import pandas as pd

def data_loader(file_name):
    """
    This function loads the data from the file
    :param file_name: name of the file
    :return: data
    """
    data = pd.read_csv(file_name)
    # describe the data
    print(data.describe().to_string())

    # write data description in a text file in output folder
    with open("output/data_description.txt", "w") as f:
        f.write(data.describe().to_string())
        # write column names before processing
        f.write("\n\nColumn names before processing:\n")

        for col in data.columns:
            f.write(col + "\n")
    return data

def data_preprocess(data, y_name = "None"):
    """
    This function preprocesses the data

    :param data: data
    :param y_name: name of the target variable
    :return: X, y

    """
    data = data.dropna(thresh = 0.8 * len(data), axis = 1)



    data = data.drop_duplicates()

    if y_name is None:
        # considering the first column as y
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:]
    else:
        y = data[y_name]
        X = data.drop(y_name, axis = 1)




    X = pd.get_dummies(X)


    cat_cols = X.select_dtypes(include = ["object"]).columns
    num_cols = X.select_dtypes(exclude = ["object"]).columns


    for col in data.columns:
        if col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
        elif col in num_cols:
            data[col] = data[col].fillna(data[col].median())


    with open("output/data_description.txt", "a") as f:
        f.write("\n\nColumn names after processing:\n")
        for col in X.columns:
            f.write(col + "\n")
        f.write("\n\nTarget variable:\n")
        f.write(y_name)

    return X, y