import matplotlib.pyplot as plt 
import numpy as np 
from multiprocessing import Pool, Manager, shared_memory, Array
from timeit import default_timer as timer
import math

from functools import partial


def find_x(entry_price, exit_price, short_long):
    #start, stop(more 0.1 for graph display), step(0.1 graph จะได้ละเอียด)
    if entry_price > exit_price:
        return np.arange(exit_price+(0.1*short_long), (entry_price+0.1), 0.1)
    else :
        return np.arange(entry_price, exit_price+(0.1*short_long), 0.1)

def convert_x(x):
    # xList = x.tolist()
    n = math.ceil(len(x)/6) #list 2D, 6 element (6 is core cpu)
    xList = [x[i:i+n] for i in range(0, len(x), n)]
    tupleX = list(zip(xList)) #convert list to tuple
    return tupleX

def inverse_calculate(x, quantity, entry_price, short_long, leverage, entry_value):
    return short_long*leverage*(quantity/entry_price - quantity/x)/entry_value*100

def normal_calculate(x, short_long, leverage, entry_price):
    return short_long*leverage*(x-entry_price)/entry_price*100

def display(chart_, x, y_normal, y_roe_inverse, roe, roe_linear, short_long):
    if chart_ == 1:
        plt.plot(x, y_normal, label = 'normal')
        plt.title('normal') 
    elif chart_ == 2:
        plt.plot(x, y_roe_inverse, label = "Inverse") 
        plt.title('Inverse') 
    else:
        plt.plot(x, y_normal, label = 'normal')
        plt.plot(x, y_roe_inverse, label = "Inverse") 
        plt.title('Inverse vs Linear') 
    
    if (short_long == -1):
        plt.xlim(max(x), min(x))
    print('ROE inverse return = %.2f%%' %(roe))
    print('ROE linear return = %.2f%%' %(roe_linear))

    plt.xlabel('price') 
    plt.ylabel('ROE%') 
    plt.legend()
    plt.grid()
    plt.show() 

if(__name__=='__main__'):
    quantity = 1000
    entry_price = 1
    exit_price = 10000
    exit_price_c = exit_price-1
    leverage = 1
    short_long = int(input('long(1) or short(-1) : '))
    chart_ = int(input('normal chart(1), inverse chart(2), all chart (3) : '))

    #define x forward(long)(1) or backward(short))-1
    x = find_x(entry_price, exit_price, short_long)

    #ROE inverse calculate
    entry_value = quantity/entry_price
    exit_value = quantity/exit_price
    profit = entry_value-exit_value
    roe = profit/entry_value*100*leverage*short_long

    #ROE linear calculate
    roe_linear = short_long*leverage*(exit_price-entry_price)/entry_price*100

    #graph y calculate sequential
    start = timer()
    y_roe_inverse = [(short_long*leverage*(quantity/entry_price - quantity/xs)/entry_value*100) for xs in x]
    # y_roe_inverse = short_long*leverage*(quantity/entry_price - quantity/x)/entry_value*100
    end = timer()
    print('ROE inverse sequential: ', (end-start))

    start = timer()
    y_normal = [(short_long*leverage*(xs-entry_price)/entry_price*100) for xs in x]
    end = timer()
    print('ROE normal sequential: ', (end-start))

    # print(type(x))

    #graph calculate parallel
    pool = Pool(processes = 6)
    tupleX = convert_x(x)
    #time
    start = timer()

    y_roe_inverse_parallel = pool.starmap(
        partial(
            inverse_calculate, 
            quantity = quantity, 
            entry_price = entry_price, 
            short_long = short_long, 
            leverage = leverage, 
            entry_value = entry_value
            ), tupleX
    )
    
    end = timer()
    print('ROE inverse parallel: ', (end-start))

    start = timer()
    y_normal_parallel = pool.starmap(
        partial(
            normal_calculate,
            entry_price = entry_price, 
            short_long = short_long, 
            leverage = leverage,
            ), tupleX
    )
    end = timer()
    print('ROE normal parallel', (end-start))

    display(chart_, x, y_normal, y_roe_inverse, roe, roe_linear, short_long)