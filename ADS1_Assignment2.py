import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import skew


def reader(file):
    '''
    It is defined to pass on the file name of the data frame, then later read
    the data frame and transpose it. Later the Original data frame & transposed
    data frame is returned
    Parameters
    ----------
    file :It holds the name of World Bank Data set which needs to be
    transposed further operations are being carried out.

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
    toppop : Only Top 5 populated countries and attributes such as Country code
    ,Indicator Name & Code is dropped

    '''
    print(df)
    pop = df.groupby(['Indicator Code']).get_group('SP.POP.TOTL')
    pop = pop.reset_index().drop(['index'], axis=1)
    value = []
    value = ['ARB', 'CEB', 'EAS', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC',
             'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'LIC', 'LMC', 'LMY', 'MEA',
             'MIC', 'NAC', 'PSS', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN',
             'TSA', 'TSS', 'UMC', 'WLD', 'EAR', 'LTE', 'EAP', 'SAS', 'OED',
             'SSA', 'PST', 'PRE', 'AFE', 'AFW', 'LAC', 'LCN', 'MNA', 'ECA']
    toppop = pop[pop['Country Code'].isin(value) == False]
    toppop = toppop.drop(columns=['Country Code', 'Indicator Name',
                                  'Indicator Code'], axis=1)
    toppop = toppop.sort_values(by=2021, ascending=False,
                                ignore_index=True).head()
    print(toppop)
    toppop.to_csv('fivetoppop.csv')
    return toppop


def stati_corr(df):
    '''
    This function is defined to plot the Correlation heat map with certain
    desired indicators of Country USA, Hence the Original Data is passed from
    main program as an arguement. Further the Data is manipulated process
    using pandas function to obatain data with desired indicator,latter
    is through passed corr() for pair wise correlation and mapped to heatmap
    using using sb.heatmap(). The Heat Map is plotted within this function.

    Parameters
    ----------
    df : This Data frame holds the Original Data

    Return
    None
    '''

    # Plotting of Correlation of heatmap
    # Storing  the Values of indicators for Country USA to new data frame.
    us_df = df.groupby(['Country Name']).get_group('United States')
    us_df = us_df.reset_index().drop(['index'], axis=1)
    # Alloting only keywords of desired indicator to an empty array Value
    value = ['SP.POP.TOTL', 'EN.ATM.CO2E.LF.KT', 'EG.USE.PCAP.KG.OE',
             'EG.FEC.RNEW.ZS', 'EG.ELC.NGAS.ZS', 'SP.POP.TOTL',
             'EN.ATM.CO2E.LF.KT', 'EG.USE.PCAP.KG.OE', 'EG.FEC.RNEW.ZS',
             'EG.ELC.NGAS.ZS', 'SP.URB.TOTL', 'SP.URB.TOTL.IN.ZS']
    # Taking out rows with certain Indicator alone  is stored in array Value
    us_id = us_df[us_df['Indicator Code'].isin(value) == True]
    us_id = us_id.drop(columns=['Country Name', 'Country Code',
                                'Indicator Code'], axis=1)
    # Transposing the Dataframe for better plotting of Correlation Heatmap.
    us_id = us_id.set_index('Indicator Name').T.reset_index()
    us_id = us_id.rename_axis(None, axis=1)
    us_id.to_csv('USind.csv')
    us_id = us_id.fillna(us_id.mean())
    print(us_id.isna().sum())
    us_id = us_id.drop(columns=['index'], axis=1)
    plt.figure(dpi=144, figsize=(10, 7))
    sb.heatmap(us_id.corr(), annot=True)
    plt.title('Correlation Heatmap  for US')
    plt.show()
    # Plotting of Correlation of heatmap of China
    # Storing  the Values of indicators for Country China to new data frame.
    ch_df = df.groupby(['Country Name']).get_group('China')
    ch_df = ch_df.reset_index().drop(['index'], axis=1)
    # Alloting only keywords of desired indicator to an empty array Value
    value = ['SP.POP.TOTL', 'EN.ATM.CO2E.LF.KT', 'EG.USE.PCAP.KG.OE',
             'EG.FEC.RNEW.ZS', 'EG.ELC.NGAS.ZS', 'SP.POP.TOTL',
             'EN.ATM.CO2E.LF.KT', 'EG.USE.PCAP.KG.OE', 'EG.FEC.RNEW.ZS',
             'EG.ELC.NGAS.ZS', 'SP.URB.TOTL', 'SP.URB.TOTL.IN.ZS']
    # Taking out rows with certain Indicator alone  is stored in array Value
    ch_id = ch_df[us_df['Indicator Code'].isin(value) == True]
    ch_id = ch_id.drop(columns=['Country Name', 'Country Code',
                                'Indicator Code'], axis=1)
    # Transposing the Dataframe for better plotting of Correlation Heatmap.
    ch_id = ch_id.set_index('Indicator Name').T.reset_index()
    ch_id = ch_id.rename_axis(None, axis=1)
    ch_id.to_csv('Chind.csv')
    ch_id = ch_id.fillna(ch_id.mean())
    print(ch_id.isna().sum())
    ch_id = ch_id.drop(columns=['index'], axis=1)
    plt.figure(dpi=144, figsize=(10, 7))
    sb.heatmap(ch_id.corr(), annot=True)
    plt.title('Correlation Heatmap  for China')
    plt.show()
    return None


# Passing the World Bank Data name as the arguements to reader function
df, dft = reader('Whole_Data.xlsx')   # The Original Data & Transposed data

# Original Data is passed to topfivepop() find the Top five populated Countries
tfp = topfivepop(df)  # Returned Dataframe contains Top 5 Populated Countries

# Storing Country name for future filtering of Data
country_names = tfp['Country Name'].values.tolist()
# Setting the Values in Table Country name as attributes
tfp = tfp.set_index('Country Name').T.reset_index()
tfp = tfp.rename_axis(None, axis=1)
tfp.rename(columns={'index': 'Year'}, inplace=True)
print(tfp.columns.values)

# Plot the Bar Graph of Top 5 countries for an interval of 10 years from 1980
years = []
years = [1980, 1990, 2000, 2010, 2020]
tfp_new = tfp.loc[tfp['Year'].isin(years)].reset_index().drop(['index'],
                                                              axis=1)

print(tfp_new)

plt.figure(figsize=(10, 8))

x_axis = np.arange(len(tfp_new['Year']))

plt.bar(x_axis - 0.05, tfp_new['China'], width=0.05, label='China')
plt.bar(x_axis + 0.0, tfp_new['India'], width=0.05, label='India')
plt.bar(x_axis + 0.05, tfp_new['Least developed countries: UN classification'],
        width=0.05, label='Least Devoloped Countries')
plt.bar(x_axis + 0.10, tfp_new['United States'], width=0.05,

        label='United States')
plt.bar(x_axis + 0.15, tfp_new['Indonesia'], width=0.05, label='Indonesia')
plt.title('Population of Top 5 Countries')
plt.xlabel('Years')
plt.ylabel('Population')
plt.legend()
plt.xticks(x_axis, tfp_new['Year'])
plt.savefig('Bar.png', dpi=300)
plt.show()


# Store the data of Desired Values in Country Name column to a new Data Frame

country_names.extend(['World', 'Japan', 'Germany'])
new_df = df[df['Country Name'].isin(country_names) == True]
new_df = new_df.reset_index().drop(['index'], axis=1)

# Plot the line Graph for CO2 Emmision from Desired Population
Carbon_Emi = new_df.groupby(['Indicator Code']).get_group('EN.ATM.CO2E.LF.KT')
Carbon_Emi = Carbon_Emi.reset_index().drop(['index'], axis=1)

Carbon_Emi = Carbon_Emi.drop(columns=['Country Code', 'Indicator Name',
                                      'Indicator Code'], axis=1)
print(Carbon_Emi)
Carbon_Emi.to_csv('Carbon_Emission.csv')

Carbon_T = Carbon_Emi.set_index('Country Name').T.reset_index()
Carbon_T = Carbon_T.rename_axis(None, axis=1)
Carbon_T.rename(columns={'index': 'Year'}, inplace=True)

# Removing the Rows containg Missing Values
Carbon_TD = Carbon_T.dropna()
plt.figure(figsize=(10, 8))
# Get current axis
plt.style.use('ggplot')
ax = plt.gca()
Carbon_TD.plot(kind='line', x='Year', y='China', ax=ax)
Carbon_TD.plot(kind='line', x='Year', y='India', ax=ax)
Carbon_TD.plot(kind='line', x='Year', y='Indonesia', ax=ax)
Carbon_TD.plot(kind='line', x='Year',
               y='Least developed countries: UN classification', ax=ax)
Carbon_TD.plot(kind='line', x='Year', y='United States', ax=ax)
Carbon_TD.plot(kind='line', x='Year', y='Japan', ax=ax)
Carbon_TD.plot(kind='line', x='Year', y='Germany', ax=ax)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.ylabel('CO2 Emission Kt')
plt.title('CO2 Emmision from Liquid Fuel from 1990-2015')
plt.savefig('line.png', dpi=400)
# show the plot
plt.show()

# Comparison of USA with Rest of the World
Carbon_T['Rest of world'] = Carbon_T['World']-Carbon_T['United States']

print(Carbon_T)

# Line Plot of the USA Vs Rest of the World
plt.figure(figsize=(18, 10), dpi=144)

plt.style.use('ggplot')
plt.plot(Carbon_T['Year'], Carbon_T['United States'], label='United States')
plt.plot(Carbon_T['Year'], Carbon_T['Rest of world'],
         label='Rest of the World')
plt.title('CO2 Emission from liquid fuel US Vs Rest of world', fontsize=20)
plt.xlabel('Years', fontsize=20)
plt.ylabel('Emmision of CO2 kt', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('Line2.png', dpi=300)
plt.show()

# Barplot for average CO2 Emission & Rest of the world
us_avg = np.mean(Carbon_T['United States'])
rest_avg = np.mean(Carbon_T['Rest of world'])
plt.figure(figsize=(10, 8))

label = []
label = ['USA', 'Rest of World']

x_axis = np.arange(len(label)-1)
plt.bar(x_axis - 0.15, us_avg, width=0.3, label='USA')
plt.bar(x_axis + 0.15, rest_avg, width=0.3, label='Rest of World')
plt.title('Average CO2 Emission of US Vs Rest of world')
plt.ylabel('CO2 Emission Kt')
plt.legend()
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.savefig('Bar2.png', dpi=300)
plt.show()


# Pie plot  USA vs Rest of World when US Marked its highest emission

print(Carbon_T.loc[Carbon_T['United States'].idxmax()])
# 1987 is year when USA Marked its highest consumption of CO2

# Transposing the Data for better plotting of Pie chart
Carbon = Carbon_T.astype(object).T
header = Carbon.iloc[0]
Carbon = Carbon.iloc[1:]
Carbon.columns = header
print(Carbon)
Carbon.to_csv('Carbon.csv')
Carbon = Carbon.reset_index(level=0)
print(Carbon.columns.values)
Carbon.rename(columns={'index': 'Country Name'}, inplace=True)
value = ['United States', 'Rest of world']
Carbon = Carbon.loc[Carbon['Country Name'].isin(value) == True]
print(Carbon)
ax = Carbon.groupby(['Country Name']).sum().plot(kind='pie', y=1978,
                                                 autopct='%1.0f%%',
                                                 shadow=True,
                                                 explode=[0.05, 0.05],
                                                 legend=True,
                                                 title='CO2 Emission of USA vs Rest of World-1978',
                                                 ylabel='', labeldistance=None)
ax.legend(bbox_to_anchor=(1, 1), loc=2)
plt.savefig('Pie.png', dpi=300)
plt.show()

# Skewness for the Given Distribution
us_skew = Carbon_T['United States']
us_skew = us_skew.dropna()
print('The Skewness of the Distribution is', skew(us_skew))


# Histogram of CO2 emission of USA
plt.figure(dpi=144)
Carbon_T.hist(column='United States')
plt.suptitle("Distribution of CO2 Emission for")
plt.savefig('hist.png', dpi=300)
plt.xlabel('The Distribution is -vely skewed by -1.083')
plt.show

# Function call for plotting  Correlation Heatmap of USA & China's Indicators
stati_corr(df)
