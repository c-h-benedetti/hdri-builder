import cv2 as ocv
import time_stamp as ts
import os

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     RECEIVES A LIST OF IMAGES AND WRITES IT AS A SEQUENCE                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def write_maps(maps, base_nom):
    ts.start_function("writer")
    for index, map in enumerate(maps):
        nom = base_nom + "_" + str(index) + ".png"
        maxi = map.max()
        if (maxi > 1.0):
            pass
        else:
            map = map * 255
        entiers = map.astype(int)
        ocv.imwrite(nom, entiers)
    ts.tour("maps written", "writer")



def rename_sequence(direc, seq_ini):
    entries = os.listdir(direc)
    detached = []

    for entry in entries: # We save only the extension, in lower case 
        temp = entry.split('.')
        temp[-1] = temp[-1].lower()
        detached.append(temp[-1])

    index = 0 # Future index of the files

    for ext, entry in zip(detached, entries):
        os.rename(direc+entry, direc+seq_ini+"-"+str(index)+"."+ext)
        index += 1



def open_sequence(direc):
    entries = os.listdir(direc)
    sequence = []

    for entry in entries:
        sequence.append(ocv.imread(direc+entry))
    
    return sequence


def profile_through_mono(c, l, sequence, file):
    val = []
    for img in sequence:
        val.append(img[c][l])
    
    f = open(file + ".plt", "w")
    for index, entry in enumerate(val):
        f.write(str(index) + "  " + str(entry)+"\n")
    f.close()
