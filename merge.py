import numpy as np
import pandas as pd

def binary_sampler(p, rows, cols):
    '''
    Sample binary random variables, used for manually creating dataset with missing values.
    Args:
        - p: probability of 1
        - rows: the number of rows
        - cols: the number of columns
        
    Returns:
        - binary_random_matrix: generated binary random matrix.
    '''
    np.random.seed(50)
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix

def FilePreprocess(file, preview_mode=False, DEMO_MODE=False):
    '''
    Read the file into pd.DataFrame, and do some preprocesses for demo mode.

    Args:
        - file: name of the input file
        - preview_mode: whether used for previewing, if True, only read the first one hundred lines
        - DEMO_MODE: whether used for demonstration, if True, do some preprocesses

    Returns:
        - df1: loaded dataframe
    '''
    if type(file) == str:
        if file[len(file)-4:] == '.csv':
            if preview_mode:
                # DEMO-MODE:
                # The reason why we need to distinguish the demo-mode is that the data file we use in the demo
                # (1) does not really contain missing data, so we need to create empty item manually
                # (2) does not contain the first line as the column name, so we add it manually
                # (3) we does not want to use the first column, so we drop it manually
                # The missing rate we use is 20%
                if DEMO_MODE:
                    df1 = pd.read_csv(file, header=None, skip_blank_lines=True, nrows=100)
                    df1.drop(columns=df1.columns[0], inplace=True)
                    df1.columns = ["Noaa distance", "Average temperature", "Minimum temperature", "Maximum temperature", "Rainfall", "Snowfall", "Dew point", "Relative humidity"]
                    # Introduce missing data
                    no, dim = df1.shape
                    np.random.seed(5)
                    data_m = binary_sampler(1-0.2, no, dim)
                    df1[data_m==0] = np.nan                    
                else:
                    df1 = pd.read_csv(file, skip_blank_lines=True, nrows=100)
                return np.round(df1, 6)
            else:
                # DEMO_MODE: Same description as above
                if DEMO_MODE:
                    df1 = pd.read_csv(file, header=None, skip_blank_lines=True)
                    df1.drop(columns=df1.columns[0], inplace=True)
                    df1.columns = ["Noaa distance", "Average temperature", "Minimum temperature", "Maximum temperature", "Rainfall", "Snowfall", "Dew point", "Relative humidity"]
                    # Introduce missing data
                    no, dim = df1.shape
                    np.random.seed(5)
                    data_m1 = binary_sampler(1-0.2,100, dim)
                    np.random.seed(5)
                    data_m2 = binary_sampler(1-0.2,no-100, dim)
                    data_m = np.concatenate((data_m1, data_m2))
                    df1[data_m == 0] = np.nan
                else:
                    df1 = pd.read_csv(file, skip_blank_lines=True)
                return np.round(df1, 6)
    elif type(file) == pd.core.frame.DataFrame:
        df1 = file  
    return np.round(df1, 6)

def MergeTwoFile(file1, file2, file1RealCol, DEMO_MODE=False):
    '''
    Merge two different files.
    Args:
        - file1: name of the first file
        - file2: name of the second file
        - file1RealCol: name of columns of first file
    Returns:
        - df: dataframe of the merged file
        - file1RealCol_Copy: name of columns of the merge dataframe
    '''
    df1 = FilePreprocess(file1, DEMO_MODE=DEMO_MODE)
    df2 = FilePreprocess(file2, DEMO_MODE=DEMO_MODE)    

    # ===============SPECIAL CASE====================
    name1 = set(df1.columns)
    name2 = set(df2.columns)
    if len(name1.intersection(name2)) == 0:
        df = pd.concat([df1, df2], axis=1)
        if file1RealCol == None:
            file1RealCol = list(df1.columns)
        file1RealCol_Copy = file1RealCol[:]
        file1RealCol_Copy.append(list(df2.columns))
        return df, file1RealCol_Copy
    # ===============================================

    # Start the Merging
    if file1RealCol == None:
        file1RealCol = list(df1.columns)
    file1RealCol_Copy = file1RealCol[:]
    for i in range(1, len(df2.columns)):
        Col2 = df2.columns[i]
        SameNameCounter = 0
        MatchFlag = False
        for j in range(1, len(file1RealCol)):
            Col1Real = file1RealCol[j]
            Col1 = df1.columns[j]
            if Col2 == Col1Real:
                SameNameCounter += 1
                # Create a dictionary for comparing
                Id_Col_Dict1 = {}
                for k in range(df1.shape[0]):
                    Id_Col_Dict1[df1.iat[k,0]] = (df1.iloc[k][Col1], k)
                # Start checking whether Col2 matches Col1
                MatchFlag = False
                for k in range(df2.shape[0]):
                    if df2.iat[k,0] in Id_Col_Dict1:
                        if str(df2.iloc[k][Col2]) == 'nan' or str(Id_Col_Dict1[df2.iat[k,0]][0]) == 'nan' or Id_Col_Dict1[df2.iat[k,0]][0] == df2.iloc[k][Col2]:
                            pass
                        else:
                            break
                    if k+1 == df2.shape[0]:
                        MatchFlag = True
                # When matching, fill the empty part
                if MatchFlag:
                    for k in range(df2.shape[0]):
                        if df2.iat[k,0] in Id_Col_Dict1:
                            if str(df2.iloc[k][Col2]) == 'nan' and str(Id_Col_Dict1[df2.iat[k,0]][0]) != 'nan':
                                df2.iloc[k,i] = Id_Col_Dict1[df2.iat[k,0]][0]
                            elif str(df2.iloc[k][Col2]) != 'nan' and str(Id_Col_Dict1[df2.iat[k,0]][0]) == 'nan':
                                df1.iloc[Id_Col_Dict1[df2.iat[k,0]][1],j] = df2.iloc[k][Col2]
                    df2.rename(columns={Col2: Col1}, inplace=True)
                    break
        if not MatchFlag:
            if SameNameCounter == 0:
                file1RealCol_Copy.append(Col2)
            else:
                file1RealCol_Copy.append(Col2)
                df2.rename(columns={Col2: Col2+"_"+str(SameNameCounter)}, inplace=True)
    df = pd.merge(df1, df2, how='outer')
    return df, file1RealCol_Copy
