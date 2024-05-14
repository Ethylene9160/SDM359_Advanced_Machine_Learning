import numpy as np

# Refers to page 216 in textbook.
def s(i):
    '''
    A sample function to be operated.
    :param i:
    :return:
    '''
    return i

def f(s):
    '''
    A sample function to be optimized.
    :param s:
    :return:
    '''
    return -s**2+4*s-3

def evolution_example1():
    k = 8
    bsc = -2333333
    bsi = 0
    for i in range(k):
        csc = f(s(i))
        if csc > bsc:
            bsi = i
            bsc = csc
    print('best i is: ', bsi)

def evolution_example2():
    k = 10
    S = [i for i in range(k)]
    best = -1
    bestscore = -2333333
    for i in range(k):
        selected = np.random.randint(k-i) # generate a random number between 0 and 9
        csc = f(s(S[selected]))
        if csc > bestscore:
            best = selected
            bestscore = csc
    print('best i is: ', best)

def evolution_example3():
    k = 10
    S = [i for i in range(k)]
    g_max = 8 # max generation
    best = -1
    bestscore = -2333333
    for g in range(g_max):
        selected = np.random.randint(k-g) # generate a random number between 0 and 9
        csc = f(s(S[selected]))
        if csc > bestscore:
            best = selected
            bestscore = csc
            S.pop(selected)
    print('best i is: ', best)


if __name__ == '__main__':
    evolution_example3()
