/*
    ESP-NOW Ping Test Slave
    Modified from Lucas Saavedra Vaz - 2024

    This sketch receives ping requests and immediately sends back pong responses.
    It automatically registers the master when receiving the first ping.
*/

#include "ESP32_NOW.h"
#include "WiFi.h"
#include <esp_mac.h>
#include <vector>

/* Definitions */
#define ESPNOW_WIFI_CHANNEL 6

/* Classes */

class ESP_NOW_Ping_Peer : public ESP_NOW_Peer {
public:
  ESP_NOW_Ping_Peer(const uint8_t *mac_addr, uint8_t channel, wifi_interface_t iface, const uint8_t *lmk) 
    : ESP_NOW_Peer(mac_addr, channel, iface, lmk) {}

  ~ESP_NOW_Ping_Peer() {
    remove();
  }

  bool add_peer() {
    if (!add()) {
      log_e("Failed to register peer");
      return false;
    }
    return true;
  }

  // Called when receiving a ping request
  void onReceive(const uint8_t *data, size_t len, bool broadcast) {
    if (len >= sizeof(PingPacket)) {
      PingPacket *packet = (PingPacket *)data;
      
      if (packet->type == PING) {
        pings_received++;
        Serial.printf("Ping from " MACSTR " (seq: %lu) - Sending pong...\n", 
                     MAC2STR(addr()), packet->sequence);
        
        // Send pong response immediately
        packet->type = PONG;
        if (send((uint8_t *)packet, sizeof(PingPacket))) {
          pongs_sent++;
        } else {
          Serial.println("  Failed to send pong!");
        }
      }
    }
  }

  void print_statistics() {
    Serial.printf("Pings received: %lu, Pongs sent: %lu\n", pings_received, pongs_sent);
  }

private:
  enum PacketType { PING = 0, PONG = 1 };
  
  struct PingPacket {
    uint8_t type;
    uint32_t sequence;
    unsigned long timestamp;
  };

  uint32_t pings_received = 0;
  uint32_t pongs_sent = 0;
};

/* Global Variables */

std::vector<ESP_NOW_Ping_Peer *> masters;

/* Callbacks */

void register_new_master(const esp_now_recv_info_t *info, const uint8_t *data, int len, void *arg) {
  // Check if this master is already registered
  for (size_t i = 0; i < masters.size(); i++) {
    if (memcmp(masters[i]->addr(), info->src_addr, 6) == 0) {
      return;  // Already registered
    }
  }

  Serial.printf("New master detected: " MACSTR "\n", MAC2STR(info->src_addr));
  Serial.println("Registering master...");

  ESP_NOW_Ping_Peer *new_master = new ESP_NOW_Ping_Peer(info->src_addr, ESPNOW_WIFI_CHANNEL, WIFI_IF_STA, nullptr);
  if (!new_master->add_peer()) {
    Serial.println("Failed to register master");
    delete new_master;
    return;
  }

  masters.push_back(new_master);
  Serial.printf("Master registered successfully (total: %zu)\n\n", masters.size());
}

/* Main */

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== ESP-NOW Ping Test Slave ===");

  // Initialize Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.setChannel(ESPNOW_WIFI_CHANNEL);
  while (!WiFi.STA.started()) {
    delay(100);
  }

  Serial.println("\nWi-Fi parameters:");
  Serial.println("  Mode: STA");
  Serial.println("  MAC Address: " + WiFi.macAddress());
  Serial.printf("  Channel: %d\n", ESPNOW_WIFI_CHANNEL);
  Serial.println("\nCopy this MAC address to the master's slave_mac[] array!");

  // Initialize ESP-NOW
  if (!ESP_NOW.begin()) {
    Serial.println("Failed to initialize ESP-NOW");
    Serial.println("Rebooting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  // Register callback for new peers
  ESP_NOW.onNewPeer(register_new_master, nullptr);

  Serial.println("\nSetup complete. Waiting for ping requests...\n");
}

void loop() {
  // Print statistics every 10 seconds
  static unsigned long last_stats = 0;
  if (millis() - last_stats >= 10000) {
    last_stats = millis();
    
    if (masters.size() > 0) {
      Serial.println("\n--- Statistics ---");
      for (size_t i = 0; i < masters.size(); i++) {
        Serial.printf("Master " MACSTR ": ", MAC2STR(masters[i]->addr()));
        masters[i]->print_statistics();
      }
      Serial.println("------------------\n");
    }
  }

  delay(10);
}