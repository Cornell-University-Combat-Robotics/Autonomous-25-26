#include "ESP32_NOW.h"
#include "WiFi.h"
#include <esp_mac.h>

/* ================= CONFIG ================= */

#define ESPNOW_WIFI_CHANNEL 6

/* ================= DATA STRUCT ================= */

// MUST match sender EXACTLY
struct Packet {
  float gr, gi, gj, gk;
  float gx, gy, gz;
  float t;
};

/* ================= PEER CLASS ================= */

class ESP_NOW_Peer_Class : public ESP_NOW_Peer {
public:
  ESP_NOW_Peer_Class(const uint8_t *mac_addr, uint8_t channel, wifi_interface_t iface, const uint8_t *lmk)
    : ESP_NOW_Peer(mac_addr, channel, iface, lmk) {}

  bool add_peer() {
    return add();
  }

  void onReceive(const uint8_t *data, size_t len, bool broadcast) {

    if (len != sizeof(Packet)) {
      Serial.printf("Bad packet size: %d\n", len);
      return;
    }

    Packet p;
    memcpy(&p, data, sizeof(Packet));

    Serial.printf(
    "{\"acc\": {\"x\": %.2f, \"y\": %.2f, \"z\": %.2f}, \"rot\": {\"r\": %.2f, \"i\": %.2f, \"j\": %.2f, \"k\": %.2f}, \"temp\": {\"t\": %.2f}}\n",
    p.gx, p.gy, p.gz,
    p.gr, p.gi, p.gj, p.gk,
    p.t
  );
  }
};

/* ================= GLOBAL ================= */

ESP_NOW_Peer_Class *master_peer = nullptr;

/* ================= CALLBACK ================= */

// Called when unknown device sends broadcast
void register_new_master(const esp_now_recv_info_t *info, const uint8_t *data, int len, void *arg) {

  if (memcmp(info->des_addr, ESP_NOW.BROADCAST_ADDR, 6) != 0) {
    return;
  }

  Serial.printf("New master: " MACSTR "\n", MAC2STR(info->src_addr));

  // Create peer
  master_peer = new ESP_NOW_Peer_Class(info->src_addr, ESPNOW_WIFI_CHANNEL, WIFI_IF_STA, NULL);

  if (!master_peer->add_peer()) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("Master registered");
}

/* ================= SETUP ================= */

void setup() {
  Serial.begin(115200);
  setCpuFrequencyMhz(80);

  WiFi.mode(WIFI_STA);
  WiFi.setChannel(ESPNOW_WIFI_CHANNEL);

  while (!WiFi.STA.started()) {
    delay(50);
  }

  Serial.println("Receiver ready");

  if (!ESP_NOW.begin()) {
    Serial.println("ESP-NOW init failed");
    ESP.restart();
  }

  ESP_NOW.onNewPeer(register_new_master, NULL);

  Serial.println("Waiting for data...");
}

/* ================= LOOP ================= */

void loop() {
  delay(1000);
}