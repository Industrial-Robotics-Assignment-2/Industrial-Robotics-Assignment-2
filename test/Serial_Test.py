import serial.tools.list_ports                          # pip install pyserial

ports = serial.tools.list_ports.comports()              # Create list of COM Ports
serialinst = serial.Serial()                            # Create Serial object for communication

portslist = []

for one in ports:
    comstring = str(one)                                # Convert port to string for printing
    portslist.append(comstring)
    print(comstring)

com = input("Select desired COM port:")                 # Ask user for desired port i.e. COM3

for i in range(len(portslist)):
    if portslist[i].startswith("COM" + str(com)):       # Check if user specified port exists
        use = "COM" + str(com)
        print(use)

serialinst.baudrate = 9600                              # Establish Baudrate. Must be the same across devices
serialinst.port = use                                   # Connect Serial object to desired port
serialinst.open()                                       # Open COM Port        

while True:
    msg = input("")                                     # Encode message for printing.
    serialinst.write(msg.encode('utf-8'))               # Serial cant use regular python strings, must be utf-8 format