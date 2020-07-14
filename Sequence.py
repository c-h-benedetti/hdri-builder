import cv2 as ocv
import numpy as np
import os
import utilities

path_tempo = ".temp/"

class Sequence:
    def __init__(self, nom="seq", seq=None, c_temp=False):
        if(seq == None): # We're not copying
            self.s_nom = nom
            self.written = []
        else:
            self.s_nom = seq.s_nom
            self.written = seq.written

        self.buffer = None
        self.readable = False
        self.head = None
        self.id = 0
        self.clear_temp = c_temp
        os.system("mkdir -p " + path_tempo)
        os.system("mkdir -p " + path_tempo + nom)


    def write_seq(self, sequence):
        self.written = utilities.write_maps_png(sequence, path_tempo + self.s_nom + "/")

    def size(self):
        return len(self.written)

    def write(self, img):
        self.written.append(utilities.write_png(img, path_tempo + self.s_nom + "/", self.id))
        self.id += 1

    def __getitem__(self, id):
        if (id >= self.size()):
            raise IndexError
        else:
            return ocv.imread(self.written[id], -1) / 255.0

    def read(self):
        pass
    
    def __iter__(self):
        self.head = iter(self.written)
        return self

    def __del__(self):
        if(self.clear_temp):
            os.system("rm -r " + path_tempo+self.s_nom)

    def __next__(self):
        temp = "None"
        del self.buffer
        self.buffer = None
        self.readable = False

        try:
            temp = next(self.head)
            self.readable = True
            self.buffer = ocv.imread(temp, -1) / 255.0
            return self.buffer

        except StopIteration:
            self.head = None
            raise StopIteration

