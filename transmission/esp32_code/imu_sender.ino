#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <WiFi.h>
#include <esp_now.h>

// --- BNO085 setup ---
Adafruit_BNO08x bno = Adafruit_BNO08x(55, 0x4A);
sh2_SensorValue_t sensorValue;

// --- ESP-NOW data struct ---
typedef struct {
  float ax, ay, az;
  float gx, gy, gz;
  float qw, qx, qy, qz;
} imu_packet_t;

imu_packet_t imuData;

// Replace with receiver ESP32 MAC (get it from Serial.println(WiFi.macAddress()))
uint8_t receiverMac[] = {0x24, 0x6F, 0x28, 0xAB, 0xCD, 0xEF};

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("ESP-NOW Send Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Success" : "Fail");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // --- Initialize IMU ---
  if (!bno.begin_I2C()) {
    Serial.println("Failed to find BNO085");
    while (1) delay(10);
  }
  bno.enableReport(SH2_ACCELEROMETER);
  bno.enableReport(SH2_GYROSCOPE_CALIBRATED);
  bno.enableReport(SH2_ROTATION_VECTOR);
  Serial.println("BNO085 ready");

  // --- Initialize WiFi & ESP-NOW ---
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
  // Poll IMU and pack struct
  if (bno.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ACCELEROMETER:
        imuData.ax = sensorValue.un.accelerometer.x;
        imuData.ay = sensorValue.un.accelerometer.y;
        imuData.az = sensorValue.un.accelerometer.z;
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        imuData.gx = sensorValue.un.gyroscope.x;
        imuData.gy = sensorValue.un.gyroscope.y;
        imuData.gz = sensorValue.un.gyroscope.z;
        break;

      case SH2_ROTATION_VECTOR:
        imuData.qw = sensorValue.un.rotationVector.real;
        imuData.qx = sensorValue.un.rotationVector.i;
        imuData.qy = sensorValue.un.rotationVector.j;
        imuData.qz = sensorValue.un.rotationVector.k;
        break;
    }
  }

  // Send latest data packet
  esp_now_send(receiverMac, (uint8_t *)&imuData, sizeof(imuData));

  delay(50); // ~20 Hz
}
