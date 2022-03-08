# WoTT-Haptic-Interface

Open and Standardized interfaces for Tactile Things via W3C Web of Things

## About The WoTT Haptic Interface

 - This project implements:

   1. A module, namely *WoTTServer*, to expose a *tactile device* as a W3C *Thing* so that W3C Web of Things clients can interact with the *tactile device* via WoT APIs
   2. A WoT client that consumes the exposed thing to communicate with *haptic jacket* (vibration modules) via WoT APIs and HapTic APIs
   
## Built With
- WoTTServer
  - python
  - wotpy
  - pytorch
  - torchvision
  - opencv
- WoT Client
	- python
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
[150.65.152.91] is the ip address of JAIST WoTT server.
Then run:
```
$ python codes/WoTTClientHapticJacketInterface.py
```
This program have been tested to interface with the GUI Visualization of the Haptic JacKet. Please modify according to your needs.

The hapticAPI module is invoked every time the Deformed Event of tactile device is detected. This event is captured and processed for communication with vibration modules using ```utils/SensingCells``` class
