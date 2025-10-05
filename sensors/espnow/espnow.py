import network
import espnow
import time

# Initialize Wi-Fi in station mode
sta = network.WLAN(network.STA_IF)
sta.active(True)
sta.disconnect() # Disconnect from any previously connected Wi-Fi networks

# Initialize ESP-NOW
e = espnow.ESPNow()
e.active(True)

print("ESP-NOW Receiver active. Waiting for messages...")

while True:
    host, msg = e.recv() # Wait to receive an ESP-NOW message
    if msg: # If a message is received (msg will be None if timeout occurs)
        print(f"Received message from MAC: {host.hex(':')} - Message: {msg.decode('utf-8')}")
        # You can add your custom logic here to process the received message
        # For example, control an LED, send data to a display, etc.
    time.sleep(0.1) # Small delay to avoid busy-waiting