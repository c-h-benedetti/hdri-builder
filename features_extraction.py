import cv2 as ocv
import numpy as np
import time_stamp as ts
import convolution as cvl
import utilities as ut

g_index = 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            EXTRACTION OF CONTRAST MAP                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The contrast map highlights the detailed areas 
# Values are mapped between 1.0 and 2.0. 
# By proceeding of that way:
#    - Areas having no interest are close to 1.0 (neutral and multiplied) 
#    - Important areas are close to 2.0 (increases the weight produced when multiplied)

def contrast_map(imIn):
    map = ocv.Laplacian(imIn, -1, ksize=3)
    map = ocv.normalize(map, None, 1.0, 2.0, ocv.NORM_MINMAX, ocv.CV_32F)
    return map

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            EXTRACTION OF SATURATION MAP                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def saturation_map(imIn):
    b, g, r = ocv.split(imIn)
    avg = (b + g + r) / 3.0

    b_sq = np.power(b - avg, 2)
    g_sq = np.power(g - avg, 2)
    r_sq = np.power(r - avg, 2)

    sum_sqs = np.sqrt((b_sq + g_sq + r_sq) / 3.0)

    s = ocv.normalize(sum_sqs, None, 1.0, 2.0, ocv.NORM_MINMAX, ocv.CV_32FC2)

    return s

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            EXTRACTION OF EXPOSURE MAP                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def well_exposedness(imIn, sigma):
    channels = [b, g, r] = ocv.split(imIn)
    output = []
    deno = pow(2 * sigma, 2)

    for channel in channels:
        c = ocv.normalize(channel, None, 0.0, 1.0, ocv.NORM_MINMAX, ocv.CV_32FC2)
        c = -1.0 * (np.power((c - 0.5), 2) / deno)
        c = np.exp(c)
        output.append(c)

    render = output[0] * output[1] * output[2]
    render = ocv.normalize(render, None, 1.0, 2.0, ocv.NORM_MINMAX, ocv.CV_32FC2)

    return render

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            COMBINING ALL THE FEATURES MAPS                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def combined_map(imIn, boost_contrast=1, boost_saturation=1, boost_exposure=1):
    global g_index
    imColor = imIn
    imGray = ocv.cvtColor(imIn, ocv.COLOR_BGR2GRAY)

    map_contrast = np.power(contrast_map(imGray), boost_contrast)
    map_saturation = np.power(saturation_map(imColor), boost_saturation)
    map_exposure = np.power(well_exposedness(imColor, 0.2), boost_exposure)

    combined = map_contrast * map_saturation * map_exposure

    map_contrast = ocv.normalize(map_contrast, None, 0.0, 255.0, ocv.NORM_MINMAX)
    map_saturation = ocv.normalize(map_saturation, None, 0.0, 255.0, ocv.NORM_MINMAX)
    map_exposure = ocv.normalize(map_exposure, None, 0.0, 255.0, ocv.NORM_MINMAX)

    combined = ocv.normalize(combined, None, 0.03, 1.0, ocv.NORM_MINMAX, ocv.CV_32FC2) # Clamped to 0.03 to avoid division by 0

    ocv.imwrite("output/contrast-"+str(g_index)+".png", map_contrast)
    ocv.imwrite("output/saturation-"+str(g_index)+".png", map_saturation)
    ocv.imwrite("output/exposure-"+str(g_index)+".png", map_exposure)
    g_index += 1

    return combined

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            PROCESSING OF FEATURES MAPS FOR A SEQUENCE OF IMAGE                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def weight_maps_sequence(sequence):
    output = []

    for img in sequence:
        output.append(combined_map(img, 1.0, 1.0, 1.0))

    return output

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#            BALANCED FEATURES MAPS                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def balance_maps(maps):
    buffer = np.zeros(maps[0].shape)
    for map in maps:
        buffer += map

    def_map = []
    for map in maps:
        def_map.append(map / buffer)

    return def_map

def assemble_layers(sequence, maps):
    buffer = np.zeros(sequence[0].shape)

    for img,map in zip(sequence, maps):
        factor = ocv.merge((map, map, map))
        buffer += (img * factor)

    print("min: "+str(buffer.min()))
    print("max: "+str(buffer.max()))

    return buffer

def blend_outputs(weights, sequence):
    laplacian = [] # Contains the Laplacian pyramid of each image from the bracketed set
    gaussian = [] # Contains the Gaussian pyramid of each weight map corresponding to images from the set
    blended_laplacian = [] # Will contain our new Laplacian pyramid, that carries blender for each level of detail

    # If there are N images in the bracketed set:
    #   - There are N pyramids in the 'gaussian' list
    #   - There are N-1 pyramids in the 'laplacian' list
    # So we will dump the last level of each of our gaussian pyramids

    for img in sequence:
        laplacian.append(cvl.laplacian_pyramid(img))
    for weight in weights:
        gaussian.append(cvl.gaussian_pyramid(weight))

    shape = sequence[0].shape # Shape of the desired canvas (== the shape of the biggest map)
    depth = len(laplacian[0]) # Number of layers of the pyramid with the lowest number of floors

    for l in range(0, depth): # 'l' allows us to traverse the same layer of each pyramid at the same time.
        result = np.zeros(shape)
        for pyra_gauss, pyra_lapla in zip(gaussian, laplacian):
            gauss_work = ocv.merge((pyra_gauss[l], pyra_gauss[l], pyra_gauss[l]))
            result += (pyra_lapla[l] * gauss_work)

        blended_laplacian.append(result)

    return blended_laplacian
            

def collapse(blended_lap):
    result = np.zeros(blended_lap[0].shape)

    for map in blended_lap:
        result += map

    result = ocv.normalize(result, None, 0.0, 255.0, ocv.NORM_MINMAX, ocv.CV_8UC3)
    return result

