import csv

#temperature = []
with open('GlobalTemperatures.csv', newline='') as csvfile:
    #data = csv.reader(csvfile, delimiter=',', quotechar='|')
    data = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    print("Date", ":", "Average Temperature")
    for row in data:
        #
        #temperature.append(row['Temperature'])
        #print("  |  ".join(row))
        #print(row['dt'], ":", row['LandAverageTemperature'])
        print(row) #Ver como converter para float

#Fazer a media mensal 
#Para trabalhar com meses ao inves de variação diaria 