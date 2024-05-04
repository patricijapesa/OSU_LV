import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

#zadatak 1

print('Broj mjerenja: ', len(data))
print('Tipovi veličina: ', data.dtypes)
print('Broj izostalih veličina: ', data.isnull().sum())
print('Broj dupliciranih veličina: ', data.duplicated().sum())
data.dropna(axis = 0)
data.drop_duplicates()
data = data.reset_index(drop = True)
data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']] = data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']].astype('category')

data_sortedbyconsumption = data.sort_values(by = 'Fuel Consumption City (L/100km)', ascending = True)
print("Najmanja gradska potrošnja: ", data_sortedbyconsumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))
print("Najveća gradska potrošnja: ", data_sortedbyconsumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3))

data_sortedbymotorsize = data[data['Engine Size (L)'].between(2.5, 3.5)]
print('Broj vozila kojima je veličina motora između 2.5 i 3.5 L: ', data_sortedbymotorsize.size)
print('Prosjecna CO2 emisija plinova za aute s odabranom veličinom motora: ', data_sortedbymotorsize['CO2 Emissions (g/km)'].mean())

audi_data = data[data['Make'] == 'Audi']
print('Broj mjerenja koji se odnosi na vozila proizvođača Audi: ', audi_data.size)
audi4cylinders = audi_data[audi_data['Cylinders'] == 4]
print('Prosječna CO2 emisija plinova Audi automobila s 4 cilindra: ', audi4cylinders['CO2 Emissions (g/km)'].mean().__round__(2))

data_groupedbycylinders = data.groupby('Cylinders')
cars_with_even_cylinders = data[data['Cylinders'] % 2 == 0].groupby('Cylinders')
print('Broj vozila s parnim brojem cilindara: ', cars_with_even_cylinders.size())
print('Prosječna CO2 emisija s obzirom na broj cilindara', data_groupedbycylinders['CO2 Emissions (g/km)'].mean().__round__(2))

diesel_cars = data[data['Fuel Type'] == 'D']
gasoline_cars = data[data['Fuel Type'] == 'X']
print('Prosječna gradska potrošnja automobila koji koriste dizel: ', diesel_cars['Fuel Consumption City (L/100km)'].mean().__round__(2))
print('Prosječna gradska potrošnja automobila koji koriste benzin: ', gasoline_cars['Fuel Consumption City (L/100km)'].mean().__round__(2))
print('Medijalne vrijednosti dizel: ', diesel_cars['Fuel Consumption City (L/100km)'].median())
print('Medijalne vrijednosti benzin: ', gasoline_cars['Fuel Consumption City (L/100km)'].median())

diesel_cars_with4cylinders = diesel_cars[diesel_cars['Cylinders'] == 4]
sorted_diesel_cars_with4cylinders= diesel_cars_with4cylinders.sort_values(by = 'Fuel Consumption City (L/100km)', ascending = True)
print('Dizel koji ima 4 cilindra i najveću gradsku potrošnju: ', sorted_diesel_cars_with4cylinders.tail(1))

cars_sortedbytransmission = data[data['Transmission'].str.startswith('M')]
print('Broj automobila s ručnim tipom mjenjača: ', cars_sortedbytransmission.size)

print(data.corr(numeric_only = True))


#zadatak 2

plt.figure()
plt.hist(data['CO2 Emissions (g/km)'], bins = 30, color = 'cyan')
plt.title('Emisija CO2 plinova')
plt.show()

fuels = {'X':'blue', 'Z':'yellow', 'D':'green', 'E':'red', 'N':'purple'}
for fuel_type, color in fuels.items():
    subset = data[data['Fuel Type'] == fuel_type]
    plt.scatter(subset['Fuel Consumption City (L/100km)'], subset['CO2 Emissions (g/km)'],
                color=color, label=fuel_type)
plt.legend(title='Fuel Type')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

data.boxplot(column = 'Fuel Consumption Hwy (L/100km)', by = 'Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Fuel Consumption Hwy (L/100km)')
plt.show()

cars_groupedbyfuel = data.groupby('Fuel Type').size()
cars_groupedbyfuel.plot(kind='bar', color = 'blue')
plt.xlabel('Tip goriva')
plt.ylabel('Broj vozila')
plt.show()

cars_bycylinders=data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
cars_bycylinders.plot(kind='bar',color='blue')
plt.xlabel('Broj cilindara')
plt.ylabel('CO2 emisija')
plt.show()