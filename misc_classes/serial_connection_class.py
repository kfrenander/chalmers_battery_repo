import serial
import serial.tools.list_ports
from time import sleep


# --- Helper function to list available ports ---
def list_serial_ports():
    """List all available serial ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found!")
    else:
        print("Available ports:")
        for port in ports:
            print(f"  - {port.device}: {port.description}")


class SerialManager:
    def __init__(self, port: str = None, baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None

    def connect(self):
        """Initialize connection to the serial device."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"Connected to {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")

    def disconnect(self):
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Serial connection closed.")

    def send_command(self, command: str):
        """Send a command (string) to the device."""
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Error: Not connected to device!")
            return

        try:
            self.serial_conn.write(command.encode())
            print(f"Sent: {command}")
        except serial.SerialException as e:
            print(f"Error sending command: {e}")

    def read_response(self) -> str:
        """Read response from the device."""
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Error: Not connected to device!")
            return ""

        try:
            response = self.serial_conn.readline().decode('utf-8').strip()
            print(f"Received: {response}")
            return response
        except serial.SerialException as e:
            print(f"Error reading response: {e}")
            return ""


if __name__ == "__main__":
    # List available ports (optional)
    list_serial_ports()

    # Replace with your actual port (e.g., "/dev/ttyUSB0")
    PORT = "/dev/ttyUSB0"
    BAUD_RATE = 9600

    # Initialize and connect
    ser_mgr = SerialManager(port=PORT, baudrate=BAUD_RATE)
    ser_mgr.connect()

    # Example: Send a command and read response
    if ser_mgr.serial_conn and ser_mgr.serial_conn.is_open:
        ser_mgr.send_command("HELLO")  # Replace with your device's command
        sleep(0.1)  # Small delay for device response
        response = ser_mgr.read_response()
        print(f"Device says: {response}")

    # Cleanup
    ser_mgr.disconnect()