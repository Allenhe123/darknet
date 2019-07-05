#!/usr/bin/env bash

set -e 

if [ -f /home/allen/myproject/darknet/allen_detect ]; then
	rm -f allen_detect
	echo 'delete allen_detect'
fi

if [ -f /usr/lib/libdarknet.so ]; then
	sudo rm -f /usr/lib/libdarknet.so
	echo 'delete /usr/lib/libdarknet.so and cp a new one'
	sudo ln -s /home/allen/myproject/darknet/libdarknet.so /usr/lib/libdarknet.so
else
	echo 'can not find /usr/lib/libdarknet.so, will cp it'
	sudo ln -s /home/allen/myproject/darknet/libdarknet.so /usr/lib/libdarknet.so
fi

g++ examples/allen_detect.cpp -Iinclude/ -Isrc/ -lopencv_core -lopencv_highgui -lopencv_imgproc -ldarknet -o allen_detect

set +e
