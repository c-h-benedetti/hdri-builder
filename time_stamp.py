import time
from enum import Enum

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     GLOBAL VARIABLES                                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

start_time = 0.0
previous = {}
laps = {}
text_size_limit = 32
round_time = 5
separator = "|"
buffer_limit = 100

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     TIME STAMP FUCTIONS                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Adds a new lap to the buffer

def add_lap(lap, id):
    global laps
    
    if(id in laps):
        laps[id].append(lap)
        pass
    else:
        laps[id] = [lap]
        pass

# Erases all saved values and considers curent time as entry of program
# Argument enables to reset the laps without modifying the starting time

def reset(all_vars=False):
    global laps
    global previous
    global start_time

    laps = {}
    previous = {}

    if(all_vars):
        start_time = time.time()


# Measures the time ellaŝed between two calls of this function, with 'txt' as log-line
# Automatic saving in a list

def start_function(id):
    previous[id] = time.time()

def tour(txt, id):
    global previous

    if(len(txt) > text_size_limit):
        print("Warning: Text too long")
        txt = txt[0:text_size_limit]
    
    missing = text_size_limit - len(txt)
    missing = " " * missing

    if(id in previous): # Deja init dans cette fonction
        elapsed = time.time() - previous[id]
        elapsed = round(elapsed, round_time)
        add_lap([(txt + missing + separator + "   "), elapsed], id)
        previous[id] = time.time()
        pass
    else:
        elapsed = 0.0
        add_lap([(txt + missing + separator + "   "), elapsed], id)
        previous[id] = time.time()
        pass

    

# Displays all 

def disp_laps():
    for key in laps.keys():
        print(key)
        for lap in laps[key]:
            print(lap[0] + str(lap[1]))
        print("\n")

# Writes the saved laps in the file specified as parameter

def write_laps(file, write_all=True):
    f = open(file, "w")

    for key in laps.keys():
        f.write("          ###  " + str(key) + "  ###\n")
        for lap in laps[key]:
            f.write(lap[0] + str(lap[1]) + "\n")
        f.write("\n\n")

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
    for cle in laps.keys():
        if(sorting == TimeSort.DEFAULT):
            pass
        elif(sorting == TimeSort.LONGER):
            laps[cle] = sorted(laps[cle], key=lambda tup: tup[1], reverse=True)
            pass
        elif(sorting == TimeSort.SHORTER):
            laps[cle] = sorted(laps[cle], key=lambda tup: tup[1])
            pass
        else:
            print("Warning: Unknown sorting parameter. No sorting applied.")
            pass



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     TIME STAMP FUCTIONS                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

start_time = time.time()
previous = {}
