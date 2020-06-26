import time
from enum import Enum

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     GLOBAL VARIABLES                                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

start_time = 0.0
previous = 0.0
laps = []
text_size_limit = 32
round_time = 5
separator = "|"
buffer_limit = 100

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     TIME STAMP FUCTIONS                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Adds a new lap to the buffer, regarding the max size of the buffer.
# Max size can be ignored

def add_lap(lap, ignore_size=False):
    global laps
    if((len(laps) >= buffer_limit) and (not ignore_size)):
        laps.pop(0)
    laps.append(lap)

# Erases all saved values and considers curent time as entry of program
# Argument enables to reset the laps without modifying the starting time

def reset(all_vars=False):
    global laps
    global previous
    global start_time

    laps = []
    previous = time.time()

    if(all_vars):
        start_time = time.time()


# Measures the time ellaŝed between two calls of this function, with 'txt' as log-line
# Automatic saving in a list

def tour(txt):
    global previous
    global laps

    if(len(txt) > text_size_limit):
        print("Warning: Text too long")
        txt = txt[0:text_size_limit]
    
    missing = text_size_limit - len(txt)
    missing = " " * missing

    elapsed = time.time() - previous
    elapsed = round(elapsed, round_time)
    add_lap([(txt + missing + separator + "   "), elapsed])
    previous = time.time()

# Displays all 

def disp_laps():
    for lap in laps:
        print(lap[0] + str(lap[1]))

# Writes the saved laps in the file specified as parameter

def write_laps(file, write_all=True):
    f = open(file, "w")
    for lap in laps:
        f.write(lap[0] + str(lap[1]) +'\n')

    if(write_all):
        s = ("\n\nTOTAL:\n    %s seconds  " % (time.time() - start_time))
        f.write(s)
    f.close()

# Returns the total time of execution

def exec_time():
    global start_time
    return time.time() - start_time

# Displays and returns the total time of execution

def disp_exec_time():
    print("--- %s seconds ---" % (time.time() - start_time))

class TimeSort(Enum):
    DEFAULT = 0
    LONGER = 1
    SHORTER = 2

def sort_laps(sorting=TimeSort.DEFAULT):
    global laps
    if(sorting == TimeSort.DEFAULT):
        pass
    elif(sorting == TimeSort.LONGER):
        laps = sorted(laps, key=lambda tup: tup[1], reverse=True)
        pass
    elif(sorting == TimeSort.SHORTER):
        laps = sorted(laps, key=lambda tup: tup[1])
        pass
    else:
        print("Warning: Unknown sorting parameter. No sorting applied.")
        pass



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     TIME STAMP FUCTIONS                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

start_time = time.time()
previous = time.time()
