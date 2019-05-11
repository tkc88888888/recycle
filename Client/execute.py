import os
import sys
import time
import warnings

import pigpio
import requests
from picamera import PiCamera

warnings.simplefilter(action='ignore', category=FutureWarning)

# import warnings
# with warnings.catch_warnings():
#    warnings.filterwarnings("ignore",category=FutureWarning)
#    import h5py

image0top = "/home/pi/image0top.jpg"
image0front = "/home/pi/image0front.jpg"
image1top = "/home/pi/image1top.jpg"
image1front = "/home/pi/image1front.jpg"

sizeTop = "/home/pi/plasticTop.jpg"
sizeFront = "/home/pi/plasticFront.jpg"

##### SIZER.PY


# import the necessary packages


def TopCamera(topimagepath):
    try:
        os.system('fswebcam -d /dev/video0 -r 352x288 --no-banner {}'.format(os.path.basename(topimagepath)))
    except:
        os.system("killall -9 fswebcam")
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
        with PiCamera() as camera:
            camera.resolution = (2592, 1944)
            camera.rotation = 180
            camera.capture(frontimagepath, resize=(864, 648))
            camera.close()
    print("saved Front Image")


def RequestPost(topimagepath, frontimagepath, attr):
    data = {"weightattr": attr}

    # print(data)

    file_list = [('image1top', ('image1top.jpg', open('image1top.jpg', 'rb'), 'image/jpg')),
                 ('image1front', ('image1front.jpg', open('image1front.jpg', 'rb'), 'image/jpg'))]

    try:
        response = requests.post('http://wastesorting.dlinkddns.com:8000/classify_image', files=file_list, data=data)
        print(response.text)
        return response.json()
        # dict = response.text
        # return dict

    except requests.exceptions.RequestException as e:
        print("Error sending reading: {}".format(e))
    """
    response = requests.post('http://wastesorting.dlinkddns.com:8000/classify_image', files=file_list, data=data)
    print(response.text)
    return response.json()
    #print(' time taken: {} '.format(datetime.datetime.now()-t1))
    #dict = response.text
    #return dict    
    """


a = 1

"""
#(A1)infrared sensor
infra=2
GPIO.setmode(GPIO.BCM)
GPIO.setup(infra,GPIO.IN)

"""

# """
# (A2)Load Cell/weight scale
EMULATE_HX711 = False

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711

    GPIO.setwarnings(False)

else:
    from emulated_hx711 import HX711


def cleanAndExit():
    print "Cleaning..."

    if not EMULATE_HX711:
        GPIO.cleanup()

    print "Bye!"
    sys.exit()


hx = HX711(23, 24)
hx.set_reading_format("MSB", "MSB")
hx.set_reference_unit(200)
hx.reset()
hx.tare()


# """


def main():
    # (B)
    while a == 1:
        # (B1)
        # if GPIO.input(infra):
        # a=1

        # (B2)
        """
        val0 = hx.get_weight(5)
        val1 = hx.get_weight(5) 
        val2 = hx.get_weight(5)
        val3 = hx.get_weight(5) 
        val4 = hx.get_weight(5) 
       
        
        val = (val0 + val1 + val2 + val3 + val4)/5
        """
        val = hx.get_weight(5)
        # val = 20

        if val <= 15:

            print ("{} is less than 15 gram. No waste detected".format(val))

            hx.power_down()
            hx.power_up()
            time.sleep(1)


        else:

            weight = round(val)
            print ("The weight of waste is {}".format(weight))

            # """
            hx.power_down()
            hx.power_up()
            # """

            TopCamera(image1top)

            time.sleep(0.75)

            FrontCamera(image1front)

            time.sleep(0.75)

            print ("sending 2 images along with weight attribute:{}".format(weight))

            response = RequestPost(image1top, image1front, weight)
            response = response['Category']
            pi = pigpio.pi()
            if 'cardboard' in response:
                print('saw cardboard')  # Bin 1 (front right) for cardboard/paper
                pi.set_servo_pulsewidth(3, 500)
                time.sleep(2)
                pi.set_servo_pulsewidth(3, 1500)
                time.sleep(1)
                pi.set_servo_pulsewidth(17, 2500)
                time.sleep(2)
                pi.set_servo_pulsewidth(17, 1500)
                time.sleep(1)

            elif 'paper' in response:
                print('saw paper')
                pi.set_servo_pulsewidth(3, 500)
                time.sleep(2)
                pi.set_servo_pulsewidth(3, 1500)
                time.sleep(1)
                pi.set_servo_pulsewidth(17, 2500)
                time.sleep(2)
                pi.set_servo_pulsewidth(17, 1500)
                time.sleep(1)

            elif 'aluminium' in response:  # Bin 2 (back right) for glass/aluminium
                print('saw aluminium')
                pi.set_servo_pulsewidth(3, 500)
                time.sleep(2)
                pi.set_servo_pulsewidth(3, 1500)
                time.sleep(1)
                pi.set_servo_pulsewidth(17, 500)
                time.sleep(2)
                pi.set_servo_pulsewidth(17, 1500)
                time.sleep(1)

            elif 'glass' in response:
                print('saw glass')
                pi.set_servo_pulsewidth(3, 500)
                time.sleep(2)
                pi.set_servo_pulsewidth(3, 1500)
                time.sleep(1)
                pi.set_servo_pulsewidth(17, 500)
                time.sleep(2)
                pi.set_servo_pulsewidth(17, 1500)
                time.sleep(1)

            elif 'plastic' in response:  # Bin 3 (front left) for plastic/tetrapak
                print('saw plastic')
                pi.set_servo_pulsewidth(3, 2500)
                time.sleep(2)
                pi.set_servo_pulsewidth(3, 1500)
                time.sleep(1)
                pi.set_servo_pulsewidth(17, 2500)
                time.sleep(2)
                pi.set_servo_pulsewidth(17, 1500)
                time.sleep(1)

            elif 'tetrapak' in response:
                print('saw tetrapak')
                pi.set_servo_pulsewidth(3, 2500)
                time.sleep(2)
                pi.set_servo_pulsewidth(3, 1500)
                time.sleep(1)
                pi.set_servo_pulsewidth(17, 2500)
                time.sleep(2)
                pi.set_servo_pulsewidth(17, 1500)
                time.sleep(1)


            elif 'trash' in response:
                pi.set_servo_pulsewidth(3, 2500)
                time.sleep(2)
                pi.set_servo_pulsewidth(3, 1500)
                time.sleep(1)
                pi.set_servo_pulsewidth(17, 500)
                time.sleep(2)
                pi.set_servo_pulsewidth(17, 1500)
                time.sleep(1)
            time.sleep(5)


main()

print("There is other error than Request Connection")
