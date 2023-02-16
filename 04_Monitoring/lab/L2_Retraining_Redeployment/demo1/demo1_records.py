import pandas as pd
from datetime import datetime

source_location = './recorddatasource/'
filename = 'recordkeepingdemo.csv'
output_location = 'records.txt'

data = pd.read_csv(source_location+filename)

dateTimeObj = datetime.now()
time_now = str(dateTimeObj.year) + '/' + str(dateTimeObj.month) + '/'+str(dateTimeObj.day)

one_record = [source_location, filename, len(data.index), time_now]
all_records = []
all_records.append(one_record) # dummy record 1
all_records.append(one_record) # dummy record 2 = record 1

# Create TXT/CSV with record info
with open(output_location, 'w') as f:
    f.write("source_location,filename,num_entries,timestamp\n")
    for record in all_records:
        for i, element in enumerate(record):
            f.write(str(element))
            if i < len(record)-1:
                f.write(",")
            else:
                f.write("\n")

# Output: records.txt
# source_location,filename,num_entries,timestamp
# ./recorddatasource/,recordkeepingdemo.csv,4,2023/2/16
# ./recorddatasource/,recordkeepingdemo.csv,4,2023/2/16