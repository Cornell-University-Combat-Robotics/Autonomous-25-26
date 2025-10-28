/*
    ESP-NOW Broadcast Master
    Lucas Saavedra Vaz - 2024

    This sketch demonstrates how to broadcast messages to all devices within the ESP-NOW network.
    This example is intended to be used with the ESP-NOW Broadcast Slave example.

    The master device will broadcast a message every 5 seconds to all devices within the network.
    This will be done using by registering a peer object with the broadcast address.

    The slave devices will receive the broadcasted messages and print them to the Serial Monitor.
*/

#include "ESP32_NOW.h"
#include "WiFi.h"

#include <esp_mac.h>  // For the MAC2STR and MACSTR macros
#include <Adafruit_BNO08x.h>

#include <math.h>


/* Definitions */

#define ESPNOW_WIFI_CHANNEL 6

// For SPI mode, we need a CS pin
#define BNO08X_CS 10
#define BNO08X_INT 

// For SPI mode, we also need a RESET 
//#define BNO08X_RESET 5
// but not for I2C or UART
#define BNO08X_RESET -1


Adafruit_BNO08x  bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

/* Classes */

// Creating a new class that inherits from the ESP_NOW_Peer class is required.

class ESP_NOW_Broadcast_Peer : public ESP_NOW_Peer {
public:
  // Constructor of the class using the broadcast address
  ESP_NOW_Broadcast_Peer(uint8_t channel, wifi_interface_t iface, const uint8_t *lmk) : ESP_NOW_Peer(ESP_NOW.BROADCAST_ADDR, channel, iface, lmk) {}

  // Destructor of the class
  ~ESP_NOW_Broadcast_Peer() {
    remove();
  }

  // Function to properly initialize the ESP-NOW and register the broadcast peer
  bool begin() {
    if (!ESP_NOW.begin() || !add()) {
      log_e("Failed to initialize ESP-NOW or register the broadcast peer");
      return false;
    }
    return true;
  }

  // Function to send a message to all devices within the network
  bool send_message(const uint8_t *data, size_t len) {
    if (!send(data, len)) {
      log_e("Failed to broadcast message");
      return false;
    }
    return true;
  }
};

/* Global Variables */

uint32_t msg_count = 0;

// Create a broadcast peer object
ESP_NOW_Broadcast_Peer broadcast_peer(ESPNOW_WIFI_CHANNEL, WIFI_IF_STA, NULL);


/* Main */

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  // Initialize the Wi-Fi module
  WiFi.mode(WIFI_STA);
  WiFi.setChannel(ESPNOW_WIFI_CHANNEL);
  while (!WiFi.STA.started()) {
    delay(100);
  }

  if (!bno08x.begin_I2C()) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }
  
  setReports();
  delay(100);

  Serial.println("ESP-NOW Example - Broadcast Master");
  Serial.println("Wi-Fi parameters:");
  Serial.println("  Mode: STA");
  Serial.println("  MAC Address: " + WiFi.macAddress());
  Serial.printf("  Channel: %d\n", ESPNOW_WIFI_CHANNEL);

  // Register the broadcast peer
  if (!broadcast_peer.begin()) {
    Serial.println("Failed to initialize broadcast peer");
    Serial.println("Reebooting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  Serial.println("Setup complete. Broadcasting messages every 5 seconds.");
}

void setReports(void) {
  Serial.println("Setting desired reports");
  if (! bno08x.enableReport(SH2_ROTATION_VECTOR)) {
    Serial.println("Could not enable vector");
  }
  if (! bno08x.enableReport(SH2_GRAVITY)) {
    Serial.println("Could not enable vector");
  }
  if (! bno08x.enableReport(SH2_GAME_ROTATION_VECTOR)) {
    Serial.println("Could not enable vector");
  }
}

static float r = 0, i = 0, j = 0, k = 0, accuracy = 0;
static float gr = 0, gi = 0, gj = 0, gk = 0;
static float gravity_x = 0, gravity_y = 0, gravity_z = 0;
void loop() {
  // Broadcast a message to all devices within the network
  if (bno08x.wasReset()) {
    Serial.print("sensor was reset ");
    setReports();
  }
  
  if (! bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  char data[1024];

  switch (sensorValue.sensorId) {
    case SH2_ROTATION_VECTOR:
      r = sensorValue.un.rotationVector.real;
      i = sensorValue.un.rotationVector.i;
      j = sensorValue.un.rotationVector.j;
      k = sensorValue.un.rotationVector.k;
      accuracy = sensorValue.un.rotationVector.accuracy;
      break;
      // quaternion_to_euler(r, i, j , k, &roll, &pitch, &yaw);
    case SH2_GRAVITY:
      gravity_x = sensorValue.un.gravity.x;
      gravity_y = sensorValue.un.gravity.y;
      gravity_z = sensorValue.un.gravity.z;
      break;
    case SH2_GAME_ROTATION_VECTOR:
      gr = sensorValue.un.gameRotationVector.real;
      gi = sensorValue.un.gameRotationVector.i;
      gj = sensorValue.un.gameRotationVector.j;
      gk = sensorValue.un.gameRotationVector.k;
      break;
  }
  
  sprintf(data, "{\"rotation\": {\"r\": %f, \"i\": %f, \"j\": %f, \"k\": %f, \"accuracy\": %f}, \"game\": {\"r\": %f, \"i\": %f, \"j\": %f, \"k\": %f}, \"accelerometer\": {\"gravity_x\": %f, \"gravity_y\": %f, \"gravity_z\": %f}  }", r, i, j, k, accuracy, gr, gi, gj, gk, gravity_x, gravity_y, gravity_z);
  Serial.printf("%s\n", data);

  if (!broadcast_peer.send_message((uint8_t *)data, sizeof(data))) {
    Serial.println("Failed to broadcast message");
  }
}
