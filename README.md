# HDRI Builder
-------------

Project in Python (OpenCV & NumPy)
**/!\ The list of features exposed bellow is not complete, it simply represents the task with the highest priority**
Ressources: https://app.milanote.com/1JLgty1M5LQ09h?p=gsA2e1Hp2IB

##Basic features of an HDRI builder:
  - Create an HDRI from a set of bracketed pictures
  - Basic noise and (motion)blur reduction
  - Ghosting removal
  - Color correction
  - Tone-mapping
  - Reading & writting files at *.hdr* format
  - Non-destructive twiking of HDRI before saving
  - Fake HDRI "color enhancement" from single picture

**The purpose of this project is to create an HDRI Builder with as uncommon/unexisting features:**

## Global features:
    - Create an animated HDRI from (360-HDRI-canvas + HDR-Videos)
    - Create a HDRI "ready-to-use", that doesn't require previous tone-mapping
    - Generation / usage LUTs

## PC Application:
    - AI boost to create HDRI from a single picture
    - Create a HDRI to .hdr or

## Mobile Application:
    - Create real HDRI images on cellphone
    - Create an HDRI from a video, given a capture protocol
    - Create 360° HDRI without 360° camera
    - Easy transfert from mobile app to PC

-------------

##Current task:
  ✑ Build Laplacian pyramid for RGB images
  *✓ Build Gaussian pyramid for RGB images*
  *✓ Build Gaussian pyramid for single-channel images*

-------------

##Current task details:

Creation of an HDRI requires to process a set of values for each image of the set.
These values will be used as weights, to mix all the differents pixels values in order to form the final image.
This way to proceed generates visible seams into our output. To get a clean mixing, usage of a Laplacian pyramid is required.
Through the multiple resolutions proposed by each floor of the pyramid, we can achieve a seamless mixing

  Before opti:
Exec time: 9.0085421436 seconds

  After opti
Exec time: 2.7509577274 seconds  
