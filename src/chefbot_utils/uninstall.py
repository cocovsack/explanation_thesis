#!/usr/bin/env python


from pyswip import Prolog, Functor, newModule, call, Variable

if __name__ == '__main__':
    prolog = Prolog()
    prolog.assertz("father(michael,john)")
    print("passed")

