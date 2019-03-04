# Time Lapse

Take timelapse pictures with our Raspberry PI and then categorize the images using Tensorflow MobileNet

## Requirements

Python 3.6.5

## Files

### capture.py

`capture.py` - script to run on your PI that will capture images for processing.  It captures 1 image per second, saves batches of 60 images once per minute, and runs for 1440 minutes (1 day).  The files are stored in `data/` and named `file00000.npy`, `file00001.npy`, etc.

### example_classify.py

`example_classify.py` - downloads mobilenet and gives detailed information about the first 60 images in file `file00000.npy`.

