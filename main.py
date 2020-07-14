import convolution as cvl
import time_stamp as ts
import features_extraction as fe
import cv2 as ocv
import utilities as ut
import numpy as np
import Sequence as sq
import tifffile as tiff

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            MAIN()                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    
    seq = ut.open_sequence("./test_lower/")
    weights = fe.weight_maps_sequence(seq)
    b_weights = fe.balance_maps(weights)
    for w in weights:
        del w
    del weights
    blended_laplacians = fe.blend_outputs(b_weights, seq)
    for w in b_weights:
        del w
    del b_weights

    rendu = fe.collapse(blended_laplacians, brightness=1.3)
    ocv.imwrite("HDRI-test-21.png", rendu)




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            TRIGGER                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


ts.tour("--- Start ---", "main")
main()
ts.tour("--- End ---", "main")

#ts.sort_laps(ts.TimeSort.LONGER)
ts.write_laps("time_stamp_new_convo.log")