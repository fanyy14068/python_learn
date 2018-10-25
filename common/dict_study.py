def dict_sort(a):
    """
    a = {0: 1, 1: 2, 5: 3, 4: 2}
    """
    sorted_by_keys = sorted(a.items())
    assert sorted_by_keys == [(0, 1), (1, 2), (4, 2), (5, 3)]

    sorted_by_values = sorted(a.items(), key=lambda x: x[1], reverse=False)
    assert sorted_by_values == [(0, 1), (1, 2), (4, 2), (5, 3)]

    sorted_keys = sorted(a)
    assert sorted_keys == [0, 1, 4, 5]


if __name__ == '__main__':
    input_dict = {0: 1, 1: 2, 5: 3, 4: 2}
    dict_sort(input_dict)
