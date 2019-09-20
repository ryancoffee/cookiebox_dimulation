#!/usr/bin/python3
import sys
import re

def main():
    if len(sys.argv)<2 :
        print('I need some files to convert to images')
        return
    for fname in sys.argv[1:]:
        f = open(fname,'r')
        i = int(0)
        for l in f.readlines():
            oname = fname + '.img'+ format(i,'05d')
            of = open(oname,'w')
            of.write(re.sub('::','\n',l))
            of.close()
            i += 1
        f.close()
    return

if __name__ == '__main__':
    main()
