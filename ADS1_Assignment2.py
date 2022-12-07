import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
#import seaborn as sns


def reader(file):
    '''
    It is defined to pass on the file name of the data frame, then later read 
    the data frame and transpose it. Later the Original data frame & transposed
    data frame is returned
    Parameters
    ----------
    file : It holds the name of World Bank Data set which needs to be transposed 
    further operations are being carried out.

    Returns
    -------
    df : Original Data Frame of the World Bank Data
    dft : Transposed Data frame
    '''
    df = pd.read_excel(file, skiprows=4)
    dft = df.transpose()
    header = dft.iloc[0]
    dft = dft.iloc[1:]
    dft.columns = header
    print(dft)
    return df, dft

def topfivepop(df):
    '''
    This function is defined to find top ten populated countries hence the 
    th original dataframe 'df' returned from reader() function is passed as an 
    arguement to current function topfivepop(),where it is grouped by indicator
    code for population and certain aggregated country datas were removed while 
    storing desired data set into a data name toppop which contains only
    top five populated countries

    Parameters
    ----------
    df : Contains original data set

    Returns
    -------
    toppop : Only Top 5 populated countries and attributes such as Country code,
    Indicator Name & Code is dropped

    '''
    print(df)
    pop=df.groupby(['Indicator Code']).get_group('SP.POP.TOTL')
    pop=pop.reset_index().drop(['index'],axis=1)
    value=[]
    value=['ARB','CEB','EAS','ECS','EMU','EUU','FCS','HIC','HPC','IBD','IBT',
       'IDA','IDB','IDX','LIC','LMC','LMY','MEA','MIC','NAC','PSS','SSF','SST',
       'TEA','TEC','TLA','TMN','TSA','TSS','UMC','WLD','EAR','LTE','EAP','SAS',
       'OED','SSA','PST','PRE','AFE','AFW','LAC','LCN','MNA','ECA']
    toppop=pop[pop['Country Code'].isin(value) == False]
    toppop=toppop.drop(columns=['Country Code','Indicator Name','Indicator Code'],
                axis=1)
    toppop=toppop.sort_values(by=2021,ascending=False, ignore_index=True).head()
    print(toppop)
    toppop.to_csv('fivetoppop.csv')
    return toppop



# Passing the World Bank Data name as the arguements in reader function
df, dft = reader('Whole_Data.xlsx') # The Original Data & Transposed data 

# Original Data is passed to find the Top five populated Countries    
tfp=topfivepop(df) # Returned Dataframe contains the Top 5 Populated Countries
tfp=tfp.set_index('Country Name').T.reset_index()
tfp.to_csv('tfp.csv')
print(tfp)



