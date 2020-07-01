import convolution as cvl
import time_stamp as ts
import features_extraction as fe
import cv2 as ocv
import utilities as ut
import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            MAIN()                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    
    seq = ut.open_sequence("./test4/")
    weights = fe.weight_maps_sequence(seq)
    weights = fe.balance_maps(weights)
    blended_laplacians = fe.blend_outputs(weights, seq)
    ut.write_maps(blended_laplacians, "blended-lap")
    rendu = fe.collapse(blended_laplacians)
    ocv.imwrite("HDRI-test-4.png", rendu)
    
    #ut.write_maps(weights, "output/weights/test-seq")
    #hdri = fe.assemble_layers(seq, weights)
    #ocv.imwrite("HDRI.png", hdri)
    
    """
    img = ocv.imread("output/weights/test-seq_6.png")
    maps0 = cvl.gaussian_pyramid(img)
    ut.write_maps(maps0, "gaussian_weight")
    """
    """
    img = ocv.imread("data/baboon.jpg", -1)
    lap = cvl.laplacian_pyramid(img)
    shp = lap[0].shape
    neutre = np.zeros(shp)
    for i,l in enumerate(lap):
        ocv.imwrite("lap-"+str(i)+".png", l)
        neutre += l

    neutre = ocv.normalize(neutre, None, 0.0, 255.0, ocv.NORM_MINMAX, ocv.CV_8UC3)
    ocv.imwrite("combined.png", neutre)
    """



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            TRIGGER                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


ts.tour("--- Start ---", "main")
main()
ts.tour("--- End ---", "main")

#ts.sort_laps(ts.TimeSort.LONGER)
ts.write_laps("time_stamp_new_convo.log")