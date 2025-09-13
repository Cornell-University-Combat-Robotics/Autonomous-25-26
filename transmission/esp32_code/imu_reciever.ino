#include <esp_now.h>
#include <WiFi.h>

typedef struct struct_message {
  float ax, ay, az;
  float gx, gy, gz;
} struct_message;

struct_message incomingData;

void OnDataRecv(const uint8_t * mac, const uint8_t *incomingDataRaw, int len) {
  memcpy(&incomingData, incomingDataRaw, sizeof(incomingData));
//   Serial.print("Accel: ");
//   Serial.print(incomingData.ax); Serial.print(", ");
//   Serial.print(incomingData.ay); Serial.print(", ");
//   Serial.print(incomingData.az);
//   Serial.print(" | Gyro: ");
//   Serial.print(incomingData.gx); Serial.print(", ");
//   Serial.print(incomingData.gy); Serial.print(", ");
//   Serial.println(incomingData.gz);
  Serial.printf("ACC: %.2f,%.2f,%.2f; GYR: %.2f,%.2f,%.2f\n", incomingData.ax, incomingData.ay, incomingData.az, incomingData.gx, incomingData.gy, incomingData.gz);
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  esp_now_register_recv_cb(OnDataRecv);
}

void loop() {
  // Nothing, all handled in callback
}
