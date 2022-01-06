'''
                            Template for Parameter File:
In this parameter file, 
(1) You have to define a class named as 'paramter()' which will be instantiated by our 
    imputation system to use as the second input of your 'main()' function defined in
    your uploaded algorithm file.
(2) You should define the value of all parameters you need in the '__init__' function 
    of class 'parameter()' as we did in the code below.
'''

class parameter():
    def __init__(self):
        self.weight = 1             # weight for the column mean value
        self.fill_value = 'NONE'    # value to be filled for non-numerical value