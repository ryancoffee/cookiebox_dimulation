#!/usr/bin/bash
nfiles=100
for ((i=0;i<$nfiles;i++))
do 
	./src/generate_sinogram_imgseg.py ./data_sinogram/streaking.${i}.${HOSTNAME} 100 &
done
