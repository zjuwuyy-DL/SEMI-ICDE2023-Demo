'''
                            Template for Algorithm File:
In this algorithm file,
(1) You have to include a function named as 'main()' which will be invoked by our
    imputation system to impute the missing data.
(2) You can freely import any kind of module and package, either written by yourself
    or provided by python.
(3) You can define any local helper function inside this file as you want.
(4) Your 'main()' function should only include two parameters: First is the input data,
    which is of the type pandas.DataFrame. If your algorithm works in other data type
    like numpy.ndarray, you can simply cast the input data to numpy.ndarray by one line
    of code np.array(dataframe). The second parameter is of type 'parameter()' which you
    are supposed to define in the Parameter file. Inside class parameter, you are able
    to encapsulate as many parameters as you may need.
(5) The output of 'main()' function should also be of type pandas.DataFrame.
'''

import pandas as pd

'''
main: This is a sample imputation function, which fill the empty item by the mean of the
      column times the user-input weight. If the column is not of numerical type, then
      fill it with user-input filling item.

Input:
      data  pandas.DataFrame        Input data to be imputed.
      args  user-defined class      It contains 'args.weight' & 'args.fill_value' two
                                    user-defined paramters.
'''
def main(data, args):
    for column in list(data.columns[data.isnull().sum() > 0]):
        try:
            mean_value = data[column].mean()
            data[column].fillna(mean_value*args.weight, inplace=True)
        except:
            data[column].fillna(value=args.fill_value, axis=0, inplace=True)
    return data
    
    
    