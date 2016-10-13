sudo xhost +
sudo nvidia-docker run --env DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix:ro -it -p 8999:8888 -p 6006:6006 -v sharedfolder:/sharefolder:shared floydhub/dl-docker:gpu bash

