import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def prepare_classic_arm_input(dataset):
    """
    MLxtend package accepts only array of arrays that is converted to dataframes
    This method converts a given dataset to categorical form by parsing data to string
    and coupling it with column name (this is done for classical ARM approaches only)
    :param dataset:
    :return:
    """
    processed = []
    for row_index, row in dataset.iterrows():
        new_row = []
        for column, value in row.items():
            new_row.append(str(column) + "__" + str(value))
        processed.append(new_row)
    return processed


def prepare_opt_arm_input(dataset):
    opt_arm_input = []
    for index, row in dataset.data.features.iterrows():
        new_row = {}
        for row_index, item in row.items():
            new_row[row_index] = row_index + "__" + str(item)
        opt_arm_input.append(pd.Series(new_row))
    new_input = pd.DataFrame(opt_arm_input)
    return new_input

def one_hot_encoding(categorical_dataset):
    te = TransactionEncoder()
    te_ary = te.fit(categorical_dataset).transform(categorical_dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df


def get_unique_values_per_column(dataset):
    transactions = dataset.data.features
    columns = list(transactions.columns)
    value_count = 0

    unique_values = {}
    for column in columns:
        unique_values[column] = []

    for transaction_index, transaction in transactions.iterrows():
        for index in range(len(list(transaction))):
            if transaction[index] not in unique_values[columns[index]]:
                unique_values[columns[index]].append(transaction[index])
                value_count += 1

    return unique_values, value_count
