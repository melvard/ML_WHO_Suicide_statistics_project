#importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



df = pd.read_csv('suicide.csv')

df.country.nunique()

df.year.sort_values().unique()

df.age.unique()

dd =df.isnull()

ss = df.groupby(['country','year']).sum()
df.groupby(['country','year']).sum()


#creating lists for every group for loops
gender_list = ['female', 'male']
ageGroup_list = ['5-14 years', '15-24 years', '25-34 years', '35-54 years',
            '55-74 years', '75+ years']
#creating new dataframe  grouped by country
df_countryGroup = df.groupby(['country']).sum()


#creating new dataframes for each country then addինգ to dictionary every data frame
Countries_df_dic ={"Armenia": df[df.country == 'Armenia'], 
                   "Turkey":df[df.country == 'Turkey'],
                   "Azerbaijan":df[df.country == 'Azerbaijan'],
                   "Georgia":df[df.country == 'Georgia'],
                   "Iran (Islamic Rep of)":df[df.country == 'Iran (Islamic Rep of)'],
                   "Germany":df[df.country == 'Germany'],
                   "Italy": df[df.country == 'Italy'],
                   "Japan": df[df.country == 'Japan'],
                   "Russian": df[df.country =='Russian Federation'],
                   "USA":df[df.country == 'United States of America']
                   }

#creating new dataframe  grouped by year
df_yearGroup = df.groupby(['year']).sum()
df_yearGroup['YEAR'] = df_yearGroup.index


#creating x independent argument (years) and dependent y argument (suicide cases)
x = df_yearGroup.iloc[:, 2:]
y = df_yearGroup.iloc[:, 0]

#create scatter with x and y and show it
plt.scatter(x,y)
plt.show()

#createing regressor for predictions
regressor = LinearRegression()
regressor.fit(x, y)


plt.scatter(x, y, color ='red')
plt.plot(x, regressor.predict(x), color= 'blue')
plt.title('World Suicide statistics')
plt.xlabel('Years')
plt.ylabel('Suicides numbers')
plt.show()

#predit for any year
y_pre = regressor.predict([[2025]])

#create empty dataframes to add countries with suicide statistics
statistics_withCoefficent = pd.DataFrame(columns=['Country', 'suicide_coefficient'])
statistics_withCases = pd.DataFrame(columns = ['Country', 'Population_number', 'suicide_cases'])


def suicide_coefficient_counter(year, m_country_df):
    '''
    Defintion for counting k coefficient for custom year and country_dateframe.
    k = suicide_number(per_year)/population(year)
    
    :returns k coefficient
    '''
    total_suicides= m_country_df[m_country_df.year == year].groupby(['age']).sum()['suicides_no'].sum()
    population = m_country_df[m_country_df.year == year].groupby(['age']).sum()['population'].sum()
    suicide_coefficient = total_suicides/population
    
    return suicide_coefficient
    
def suicide_cases_in_population(year, population_for_cases, m_country_df):
    total_suicides= m_country_df[m_country_df.year == year].groupby(['age']).sum()['suicides_no'].sum()
    population = m_country_df[m_country_df.year == year].groupby(['age']).sum()['population'].sum()
    k = population/population_for_cases
    print(total_suicides, population, k)
    return total_suicides/k

def predict_and_createplot_for_country(m_counrty_df,gender_list, ageGroup_list, year):
    country_name = m_counrty_df['country'].iloc[0]
    missing_val_count_by_column = (m_counrty_df.isnull().sum())
    #print(missing_val_count_by_column[missing_val_count_by_column > 0])
    m_counrty_df = m_counrty_df.dropna(axis = 0)  # the lines has null values are deleted
    m_counrty_df.head()
    
    x = np.array(m_counrty_df.loc[:,'year']*1.0).reshape(-1,1)
    y = np.array(m_counrty_df.loc[:,'suicides_no'],dtype = np.float32).reshape(-1,1)
    
    #Scatter Plot
    plt.figure(figsize = [10,10])
    plt.scatter(x=x,y=y)
    plt.xlabel('Year')
    plt.ylabel('Suicides number')
    plt.show()
    
    reg = LinearRegression()
    predict_space = np.linspace(min(x), max(x)).reshape(-1,1)  
    predicition_forThisCountry = 0;
    for sex in gender_list:
        for age_group in ageGroup_list:
                df_withOneGender = m_counrty_df[m_counrty_df['sex'] == sex]
                data_sex = df_withOneGender[df_withOneGender['age'] == age_group]
                x_sex = np.array(data_sex.loc[:,'year']).reshape(-1,1)
                y_sex = np.array(data_sex.loc[:,'suicides_no']).reshape(-1,1)
                reg.fit(x_sex,y_sex)                                              
                predicted = reg.predict(predict_space)
                print(country_name,"\t| ",sex,"\t| ", age_group, '\t| R^2 Score: ', reg.score(x_sex,y_sex))                       
                plt.plot(predict_space, predicted, color = 'black', linewidth = 2)
                plt.scatter(x_sex,y_sex)
                plt.title(country_name+" - " + sex + " " + age_group)
                plt.xlabel('Year')
                plt.ylabel('Suicides number')
                plt.show()
                predicition_forThisCountry += reg.predict([[year]])

    print(country_name +f"---Suicides in {year} will be", predicition_forThisCountry)
    print("\033[1;31;43m", f"y = {reg.intercept_} + {reg.coef_}X","\033[1;37;0m")

    
for country_name in Countries_df_dic:
    country_df = Countries_df_dic[country_name]#get country dataframe of dic with key
    predict_and_createplot_for_country(country_df, gender_list, ageGroup_list, 2021)
    
    
    #using definition for counting k coefficient of suicide per year
    choosen_year_for_parsing = 1991;
    population_number = 100000;
    k_suicide = suicide_coefficient_counter(choosen_year_for_parsing, country_df) 
    statistics_withCoefficent = statistics_withCoefficent.append({'Country': country_name,
                                    'suicide_coefficient':k_suicide,
                                    'year':choosen_year_for_parsing},
                                   ignore_index=True)
    
    
    #using definition for counting suicide cases in custom population 
    suicide_cases = suicide_cases_in_population(choosen_year_for_parsing, population_number, country_df)
    statistics_withCases = statistics_withCases.append({'Country': country_name,
                                    'Population_number': population_number,
                                    'suicide_cases': suicide_cases,
                                    'year':choosen_year_for_parsing},
                                   ignore_index=True)
  

#Creating suicide with coefficient statistics plot 
import math

stat = statistics_withCoefficent
plt.rcdefaults()
fig, ax = plt.subplots()
Countries = stat.iloc[:,0]
y_pos = np.arange(len(Countries))
performance = stat.iloc[:,1]
ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(Countries)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Suicide coefficient')
ax.set_title('Suicide coefficient for countries in'+" "+str(stat.year[1]))
plt.show()


#Creating suicide with cases statistics plot 
stat = statistics_withCases
plt.rcdefaults()
fig, ax = plt.subplots()
Countries = stat.iloc[:,0]
y_pos = np.arange(len(Countries))
performance = stat.iloc[:,2]
ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(Countries)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Suicide cases')
ax.set_title("In "+str(stat.Population_number[1])+' people Suicide cases for countries in'+" "+str(stat.year[3]))
i = 0
for suicide_cases in stat.suicide_cases:
    ax.text(suicide_cases,
            i,
            str(np.around(suicide_cases, 2) if math.isnan(suicide_cases) or( suicide_cases<1 )else int(suicide_cases)),
            fontsize = 15,
            verticalalignment = "center")
    i = i+1
plt.show()

#predict example (just change country key name of dictionary to predict and create plots)
predict_and_createplot_for_country(Countries_df_dic['Russian'], gender_list, ageGroup_list, 2021)

