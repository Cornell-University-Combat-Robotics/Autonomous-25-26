#include "ESP32_NOW.h"
#include "WiFi.h"
#include <esp_mac.h>
#include <Adafruit_BNO08x.h>

/* ================= CONFIG ================= */

#define ESPNOW_WIFI_CHANNEL 6
#define SEND_INTERVAL_MS 20   // 50 Hz

/* ================= IMU ================= */

#define BNO08X_RESET -1
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

/* ================= DATA STRUCT ================= */

// Compact binary packet (much faster + cooler)
struct Packet {
  float gr, gi, gj, gk;
  float gx, gy, gz;
  float t;
};

/* ================= ESP-NOW ================= */

class ESP_NOW_Broadcast_Peer : public ESP_NOW_Peer {
public:
  ESP_NOW_Broadcast_Peer(uint8_t channel, wifi_interface_t iface, const uint8_t *lmk)
    : ESP_NOW_Peer(ESP_NOW.BROADCAST_ADDR, channel, iface, lmk) {}

  ~ESP_NOW_Broadcast_Peer() {
    remove();
  }

  bool begin() {
    if (!ESP_NOW.begin() || !add()) {
      log_e("ESP-NOW init failed");
      return false;
    }
    return true;
  }

  bool send_message(const uint8_t *data, size_t len) {
    return send(data, len);
  }
};

ESP_NOW_Broadcast_Peer broadcast_peer(ESPNOW_WIFI_CHANNEL, WIFI_IF_STA, NULL);

/* ================= GLOBAL STATE ================= */

static float gr = 0, gi = 0, gj = 0, gk = 0;
static float gx = 0, gy = 0, gz = 0;
static float t = 0;

uint32_t lastSend = 0;

/* ================= SETUP ================= */

void setReports() {
  Serial.println("Setting IMU reports");

  // 100 Hz reports 
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 10000)) {
    Serial.println("Failed to enable rotation vector");
  }

  if (!bno08x.enableReport(SH2_GRAVITY, 10000)) {
    Serial.println("Failed to enable gravity");
  }
}

void setup() {
  Serial.begin(115200);

  // Lower CPU frequency 
  setCpuFrequencyMhz(80);

  // WiFi setup
  WiFi.mode(WIFI_STA);
  WiFi.setChannel(ESPNOW_WIFI_CHANNEL);

  // Lower TX power (BIG heat reduction)
  WiFi.setTxPower(WIFI_POWER_8_5dBm);

  while (!WiFi.STA.started()) {
    delay(50);
  }

  Serial.println("WiFi ready");

  // IMU init
  if (!bno08x.begin_I2C()) {
    Serial.println("BNO08x not found");
    while (1) delay(10);
  }

  setReports();
  delay(100);

  // ESP-NOW init
  if (!broadcast_peer.begin()) {
    Serial.println("ESP-NOW failed, rebooting...");
    delay(3000);
    ESP.restart();
  }

  Serial.println("Setup complete");
}

/* ================= LOOP ================= */

void loop() {

  // Handle IMU reset
  if (bno08x.wasReset()) {
    Serial.println("IMU reset");
    setReports();
  }

  // Read sensor (non-blocking)
  if (bno08x.getSensorEvent(&sensorValue)) {

    switch (sensorValue.sensorId) {

      case SH2_GAME_ROTATION_VECTOR:
        gr = sensorValue.un.gameRotationVector.real;
        gi = sensorValue.un.gameRotationVector.i;
        gj = sensorValue.un.gameRotationVector.j;
        gk = sensorValue.un.gameRotationVector.k;
        break;

      case SH2_GRAVITY:
        gx = sensorValue.un.gravity.x;
        gy = sensorValue.un.gravity.y;
        gz = sensorValue.un.gravity.z;
        break;
    }
  }

  // Throttle transmission rate
  if (millis() - lastSend < SEND_INTERVAL_MS) return;
  lastSend = millis();

  Packet p = {gr, gi, gj, gk, gx, gy, gz, t};

  // Send
  if (!broadcast_peer.send_message((uint8_t*)&p, sizeof(p))) {
    Serial.println("Send failed");
  }

  // print every second 
  static uint32_t lastPrint = 0;
  if (millis() - lastPrint > 1000) {
    lastPrint = millis();
    t = temperatureRead();
    Serial.printf("Temp: %.2f | gx: %.2f gy: %.2f gz: %.2f\n",
              t, gx, gy, gz);
  }
}