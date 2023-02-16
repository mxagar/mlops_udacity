import os
import pandas as pd

# Define local directories to look in 
directories=['/udacity1/','/udacity2/']
# Instantiate empty dataframe: PE ratio, Stock price
final_df = pd.DataFrame(columns=['peratio','price'])

for directory in directories:
    # Files in directory
    filenames = os.listdir(os.getcwd()+directory)
    for each_filename in filenames:
        current_df = pd.read_csv(os.getcwd()+directory+each_filename)
        # Append dataframe + reset index!
        final_df = final_df.append(current_df).reset_index(drop=True)

# Now, we could do some cleaning...
final_df.drop_duplicates().reset_index(drop=True)

# Persist aggregated dataframe
final_df.to_csv('result.csv', sep=',', header=True, index=False)
