# encoding:utf-8
import csv

head = ['name', 'class', 'age']
data = [('张三', 1, 20), ('李四', 2, 21)]


def csv_write_fun():
    with open('csv_test_file.csv', 'w', newline='', encoding='utf-8') as f:
        # 如果加上encoding='utf-8',在windows用excel打开可能会显示成乱码，但是记事本或者notepad都不会，猜想是execl的毛病
        csv_writer = csv.writer(f)
        csv_writer.writerow(head)
        csv_writer.writerows(data)


if __name__ == '__main__':
    csv_write_fun()
