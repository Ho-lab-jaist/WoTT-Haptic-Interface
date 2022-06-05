# WoTT-Haptic-Interface

Open and Standardized interfaces for Tactile Things via W3C Web of Things

## About The WoTT Haptic Interface

 - This project implements:

A WoT client that consumes the exposed thing to communicate with *haptic jacket* (vibration modules) via WoT APIs and hapticAPI
   
## Built With
- WoT Client
	- wotpy
	- hapticAPI

## Getting Started
Please make sure to add the domain name of the server machine to your local computer (Linux):
```
$ sudo nano /etc/hosts
```
Then add:
```
$ 150.65.152.91  	ho-lab
```
[150.65.xxx.xx] is the ip address of JAIST WoTT server.
Then run:
```
$ cd codes/
```
```
python WoTTClientHapticJacketInterface.py http://150.65.152.91:9090/tactilesensor-5fd1037e-1c8c-df9e-7136-ebfb938fc625
```
where ```http://150.65.152.91:9090/tactilesensor-5fd1037e-1c8c-df9e-7136-ebfb938fc625``` is the url for WoTT application that exposes property affordances of TacLink device. This program have been tested to interface with the GUI Visualization of the Haptic JacKet. Please modify according to your needs.

The hapticAPI module is invoked every time the Deformed Event of tactile device is detected. This event is captured and processed for communication with vibration modules using ```utils/SensingCells``` class
