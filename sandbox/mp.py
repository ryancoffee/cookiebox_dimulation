from multiprocessing import Process, freeze_support, set_start_method, Pool
import os

def foo():
    print('hello')

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello {}'.format(name))

def f3(name,number,string):
    info('function f')
    print('hello {}\tnumber: {}\tstring {}'.format(name,number,string))

if __name__ == '__main__':
    nimages=10
    info('main line')
    p = []
    p.append(Process(target=f, args=('bob',)))
    p.append(Process(target=f3, args=('alice',4,'HAHAHAHA')))
    p.append(Process(target=f, args=('ryan',)))

    for i in range(3):
        p[i].start()

    for i in range(3):
        p[i].join()

    pool = Pool(4)
    thistuplelist = [('ryan',1,'T'),('ave',2,'E'),('omar',3,'N'),('audrey',4,'S'),('matt',5,'O'),('taryn',6,'R'),('ruaridh',7,'F'),('kareem',8,'L')]
    pool.starmap(f3, thistuplelist)
    
