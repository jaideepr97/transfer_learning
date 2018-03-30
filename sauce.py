from multiprocessing import Process
import os

def info(title):
    print (title)
    print ('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print ('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(names):
    info('function f')
    for name in names:
    	print ('hello', name)           

if __name__ == '__main__':
    print("HIIIIIIIIIIIII")
    info('main line')
    names= ['bob','matt','james','dwayne']
    for i in range (0,len(names)):
    	p = Process(target=f, args=(names,))
    	p.start()
    	p.join()
