#!/usr/bin/env bash

set -e 

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
echo "current dir: $SCRIPTPATH"

cd $SCRIPTPATH

if [ -f allen_detect ]; then
	rm -f allen_detect
	echo 'delete allen_detect'
fi

if [ -f /usr/lib/libdarknet.so ]; then
	sudo rm -f /usr/lib/libdarknet.so
	echo 'delete /usr/lib/libdarknet.so and cp a new one'
	sudo ln -s $SCRIPTPATH/libdarknet.so /usr/lib/libdarknet.so
else
	echo 'can not find /usr/lib/libdarknet.so, will cp it'
	sudo ln -s $SCRIPTPATH/libdarknet.so /usr/lib/libdarknet.so
fi

g++ examples/allen_detect.cpp -Iinclude/ -Isrc/ -L/usr/lib/x86_64-linux-gnu/ -lopencv_core -lopencv_highgui -lopencv_imgproc -ldarknet -o allen_detect

if [ ! -f yolov3.weights ]; then
	echo "yolov3.weights is not exist, downloading it ..."
	wget https://pjreddie.com/media/files/yolov3.weights
fi

set +e
