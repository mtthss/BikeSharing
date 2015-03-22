
import csv
import timestring
import numpy as np


with open("../../Competition4Shared/Data/ground_truth_hour.csv") as f1:

    file_reader = csv.reader(f1, delimiter=",")
    file_reader.next()

    day_hour_avg = np.zeros((7*24))
    day_hour_counts = np.zeros((7*24))

    for row in file_reader:

        string_hour = row[5] if int(row[5])>10 else "0"+row[5]
        string_time = string_hour+":"+"00:00"
        string_date = row[1] + " " + string_time
        date = timestring.Date(string_date)
        demand = int(row[16])

        if date.day < 20 or True:

            day_hour_avg[(date.weekday-1)*24+date.hour] += demand
            day_hour_counts[(date.weekday-1)*24+date.hour] += 1

    avg = day_hour_avg / day_hour_counts

datetime = []
pred = []

with open("../../Competition4Shared/predictions/prediction_-1991062476.csv") as f1:

    file_reader = csv.reader(f1, delimiter=",")
    file_reader.next()

    day_hour_avg = np.zeros((7*24))
    day_hour_counts = np.zeros((7*24))

    for row in file_reader:

        datetime.append(row[0])
        date = timestring.Date(row[0])
        demand = int(row[1])
        pred.append(int(round(0.95*float(demand)+0.05*day_hour_avg[(date.weekday-1)*24+date.hour])))

print len(datetime), len(pred)


with open('../data/prediction_-7670133672.csv', 'wb') as csvout:

    write_out = csv.writer(csvout, delimiter = ',')
    write_out.writerow(['datetime','count'])

    for element in xrange(len(datetime)):
        write_out.writerow([datetime[element],pred[element]])