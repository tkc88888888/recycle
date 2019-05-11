import time

import pigpio

pi = pigpio.pi() # Connect to local Pi.

#"""

pi.set_servo_pulsewidth(3, 500)
time.sleep(2)
pi.set_servo_pulsewidth(3, 1500)
time.sleep(1)
pi.set_servo_pulsewidth(3, 2500)
time.sleep(1)
pi.set_servo_pulsewidth(3, 1500)
time.sleep(1)
#pi.set_servo_pulsewidth(17, 1500)
#time.sleep(0.5)


#"""
pi.set_servo_pulsewidth(17, 500)
time.sleep(2)
pi.set_servo_pulsewidth(17, 1500)
time.sleep(1)
pi.set_servo_pulsewidth(17, 2500)
time.sleep(2)
pi.set_servo_pulsewidth(17, 1500)
time.sleep(1)
#pi.set_servo_pulsewidth(17, 1500)
#time.sleep(0.5)
#"""
# switch servo off
pi.set_servo_pulsewidth(17, 0);

pi.stop()
