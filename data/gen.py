import itertools
import csv

def save_tests(file, tests, raws = None):
    with open(file, 'w+') as f:
        w = csv.writer(f)
        for row in tests:
            w.writerow(row)

def gen_xor():
    tests = []
    for i in itertools.product(range(2), repeat = 2):
        tests.append([i[0], i[1], i[0] ^ i[1]])

    save_tests('./csv/xor.csv', tests)


def gen_gates():
    tests = [] 
    for i in itertools.product(range(2), repeat = 2):
        for j in '&|^':
            tests.append([i[0], i[1], j, eval(f'{i[0]}{j}{i[1]}')])

    save_tests('./csv/gates.csv', tests)

if __name__ == '__main__':
    #gen_xor()
    gen_gates()
