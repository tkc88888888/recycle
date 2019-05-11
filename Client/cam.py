import os

from picamera import PiCamera

image1top = "/home/pi/image0top.jpg"
image1front = "/home/pi/image0front.jpg"


def TopCamera(topimagepath):
    try:
        os.system('fswebcam -d /dev/video0 -r 352x288 --no-banner {}'.format(os.path.basename(topimagepath)))
    except:
        os.system("killall -9 fswebcam")
        os.system("killall -9 python")
        os.system("killall -9 ps")
        os.system('fswebcam -d /dev/video0 -r 352x288 --no-banner {}'.format(os.path.basename(topimagepath)))
    print("Saved Top Image")


def FrontCamera(frontimagepath):
    try:
        with PiCamera() as camera:
            camera.resolution = (2592, 1944)
            camera.rotation = 180
            camera.capture(frontimagepath, resize=(864, 648))
            camera.close()

    except:
        os.system("killall -9 python")
        os.system("killall -9 ps")
        with PiCamera() as camera:
            camera.resolution = (2592, 1944)
            camera.rotation = 180
            camera.capture(frontimagepath, resize=(864, 648))
            camera.close()
    print("saved Front Image")


def main():
    TopCamera(image1top)
    FrontCamera(image1front)


main()
