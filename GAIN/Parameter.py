'''
Parameter.py:

This is the file which defines a class of parameter for the main functon.
'''

class parameter():
    def __init__(self):
        self.batch_size = 128
        self.hint_rate = 0.9
        self.alpha = 10
        self.iterations = 2
        self.epoch = 1
        self.guarantee = 0.95
        self.thre_value = 0.001
        self.initial_value = 20000
        self.epsilon = 1.4
        self.value = 2
        self.s_miss = 1