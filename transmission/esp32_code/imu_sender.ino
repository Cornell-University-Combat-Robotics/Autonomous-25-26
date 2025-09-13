#include <esp_now.h>
#include <WiFi.h>
#include <Wire.h>
#include "MPU6050.h"   // or whichever IMU library you use

MPU6050 imu;

typedef struct struct_message {
  float ax, ay, az;
  float gx, gy, gz;
} struct_message;

struct_message imuData;

// Receiver MAC (replace with your receiver’s MAC from `WiFi.macAddress()`)
uint8_t receiverMac[] = {0x24, 0x6F, 0x28, 0xAB, 0xCD, 0xEF};

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("Send Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Success" : "Fail");
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  imu.initialize();
  if (!imu.testConnection()) {
    Serial.println("IMU connection failed!");
    while (1);
  }

  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_send_cb(OnDataSent);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }
}

void loop() {
  imu.getMotion6(&imuData.ax, &imuData.ay, &imuData.az, &imuData.gx, &imuData.gy, &imuData.gz);
  esp_err_t result = esp_now_send(receiverMac, (uint8_t *) &imuData, sizeof(imuData));
  if (result == ESP_OK) {
    Serial.println("Data sent successfully");
  } else {
    Serial.println("Error sending data");
  }
  delay(50);  // ~20 Hz
}
