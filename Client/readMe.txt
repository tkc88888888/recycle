For Raspberry:

1)cam.py takes image0top and image0front to later compare with image1top and image1front respectively.
2)pigpiod servo.py test 2 servomotors that turn 90 degree one side, return to original position, then 90 deg the other side.
3)hx711.py and emulated_hx711.py are used to measure weight
4)sizer.py used to compare 2 images and output the dimension info.
5)imgflask.py uses Flask to request.POST fake data to get prediction from remote server.
6)execute.py is like imgflask.py but with full execution from 1-4 to get images and attribute(weight) and then POST it to get prediction result.
