import time
import picamera
import numpy as np

camera = picamera.PiCamera()

pic_files = 60*24 # one day
pics_per_file = 60 # 1 pic per sec , 60 sec per file

camera.resolution = (224,224)

lapse = np.zeros((pics_per_file,224,224,3), dtype=np.uint8)
single = np.zeros((224,224,3), dtype=np.uint8)

print("Camera warmup starting")
time.sleep(5)
print("Camera warmup complete")
_ = camera.capture(single, 'rgb')

last_pic_time = time.time()

for file_num in range(1,pic_files+1) :
    for i in range(pics_per_file) :
        camera.capture(single, 'rgb')
        lapse[i] = single

        if i == pics_per_file - 1 :
            file="lapse/file"+str(file_num).zfill(5)
            np.save(file=file,arr=lapse)
            print("Saved file",file)

        elapse_pic_time = time.time() - last_pic_time
        time.sleep(1 - elapse_pic_time)
        last_pic_time += 1.0
