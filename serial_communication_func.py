import serial.tools.list_ports                                       # pip install pyserial

class SerialCom():

    ports = serial.tools.list_ports.comports()                       # Create list of COM Ports
    serialinst = serial.Serial()                                     # Create Serial object for communication

    portslist = []

    def __init__(self, baudrate):

        for one in self.ports:
            comstring = str(one)                                     # Convert port to string for printing
            self.portslist.append(comstring)
            print(comstring)

        com = input("Select desired COM port:")                      # Ask user for desired port i.e. COM3

        for i in range(len(self.portslist)):
            if self.portslist[i].startswith("COM" + str(com)):       # Check if user specified port exists
                use = "COM" + str(com)
                print(use)

        self.serialinst.baudrate = baudrate                          # Establish Baudrate. Must be the same across devices
        self.serialinst.port = use                                   # Connect Serial object to desired port
        self.serialinst.open()                                       # Open COM Port

    def read(self):
        data = self.serialinst.readline().decode().strip()
        if data:
            return data
        
    def write(self, msg):
        self.serialinst.write(str(msg).encode('utf-8'))              # Encode message for printing.
                                                                     # Serial cant use regular python strings, must be utf-8 format

if __name__ == "__main__":

    serial = SerialCom(9600)

    while True:
        receive = serial.read()

        if receive:
            print(receive)