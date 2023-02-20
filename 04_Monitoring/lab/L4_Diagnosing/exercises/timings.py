
import os
import timeit
import numpy as np

def ingestion_timing():
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing=timeit.default_timer() - starttime
    return timing

def training_timing():
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing=timeit.default_timer() - starttime
    return timing

def measure_and_save_timings():
    ingestion_timings=[]
    training_timings=[]
    
    for idx in range(20):
        ingestion_timings.append(ingestion_timing())
        training_timings.append(training_timing())
    
    final_output=[]
    final_output.append(np.mean(ingestion_timings))
    final_output.append(np.std(ingestion_timings))
    final_output.append(np.min(ingestion_timings))
    final_output.append(np.max(ingestion_timings))
    final_output.append(np.mean(training_timings))
    final_output.append(np.std(training_timings))
    final_output.append(np.min(training_timings))
    final_output.append(np.max(training_timings))
    
    return final_output
    
print(measure_and_save_timings())
# [0.7203975, 0.21127245305744852, 0.6460763329999999, 1.6362521250000002, 1.415318122849999, 0.27338612631866893, 1.2490636669999944, 2.543399083]
