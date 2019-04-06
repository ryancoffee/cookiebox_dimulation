from multiprocessing import Process, Array
import scipy
import numpy

def f(a):
    a[0] = -a[0]
    a[1] = 2*a[1]

if __name__ == '__main__':
    # Create the array
    N = int(10)
    unshared_arr = scipy.rand(N)
    a = Array('d', unshared_arr)
    print("Originally a =",a[:4])

    # Create, start, and finish the child process
    p = Process(target=f, args=(a,))
    p.start()
    p.join()

    # Print out the changed values
    print("Now a = " ,a[:4])

    b = numpy.frombuffer(a.get_obj())

    b[0] = 10.0
    print(b)
    print(a[:4])
