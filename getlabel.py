import pandas as pd

filepath = 'Weather.csv'
df = pd.read_csv(filepath, skip_blank_lines=True)
df1 = df.iloc[:,:4]
df2 = df.iloc[:,4:]
df1.to_csv('Weather_Set1.csv', index=False)
df2.to_csv('Weather_Set2.csv', index=False)


# GETLABEL_FLAG = True

# if GETLABEL_FLAG:
#     filepath = "/home/r740/YYWu/1.EDIT-GAIN/Weather.csv"
#     df = pd.read_csv(filepath, header=None, skip_blank_lines=True)
#     df1 = df.iloc[:,0]
#     df2 = df.iloc[:,1:]
#     df1.to_csv('Label.csv', index=False, header=['Label'])
#     df2.to_csv('Weather_Original.csv', index=False, header=["Noaa distance", "Average temperature", "Minimum temperature", "Maximum temperature", "Rainfall", "Snowfall", "Dew point", "Relative humidity"])
# else:
#     filepath = 'Label.csv'
#     df = pd.read_csv(filepath, skip_blank_lines=True, nrows=10)
#     print(df)
