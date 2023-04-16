import csv

population = []
with open('countries.csv', newline='', encoding='utf-8-sig') as csvfile:
    data = csv.DictReader(csvfile)
    for r in data:
        print("Country", ":", "Capital")
        # append values from population column to population list
        population.append(r['Population'])
        # displaying specific columns (Country and Capital)
        print(r['Country'], ":", r['Capital'])
    # display the population list
    print(population)