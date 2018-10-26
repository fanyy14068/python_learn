import time
import pandas as pd


def format_time_with_timezone(column):
    """
    :param column: [timestring1, timestring2, ...], such as ['2016-10-10',]
    :return:
    """
    return column.map(lambda x: pd.Timestamp(x))


def format_time_for_timestamp(column, unit='s'):
    """
    :param column: [timestamp1, timestamp2, ...], such as [1477238400, ]
    :return:
    """
    return column.map(lambda x: pd.to_datetime(x, unit=unit))

