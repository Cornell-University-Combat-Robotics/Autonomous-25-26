/*
    ESP-NOW Ping Test Master
    Modified from Lucas Saavedra Vaz - 2024

    This sketch measures round-trip latency (ping) between master and slave devices.
    The master sends ping requests and measures the time until it receives a response.
*/

#include "ESP32_NOW.h"
#include "WiFi.h"
#include <esp_mac.h>

/* Definitions */
#define ESPNOW_WIFI_CHANNEL 6
#define PING_INTERVAL 1000  // Send ping every 1 second
#define PING_TIMEOUT 500    // Consider ping failed after 500ms

/* Classes */

class ESP_NOW_Ping_Peer : public ESP_NOW_Peer {
public:
  ESP_NOW_Ping_Peer(const uint8_t *mac_addr, uint8_t channel, wifi_interface_t iface, const uint8_t *lmk) 
    : ESP_NOW_Peer(mac_addr, channel, iface, lmk) {}

  ~ESP_NOW_Ping_Peer() {
    remove();
  }

  bool begin() {
    if (!add()) {
      log_e("Failed to register peer");
      return false;
    }
    return true;
  }

  // Called when receiving a pong response
  void onReceive(const uint8_t *data, size_t len, bool broadcast) {
    if (len >= sizeof(PingPacket)) {
      PingPacket *packet = (PingPacket *)data;
      
      if (packet->type == PONG) {
        unsigned long rtt = micros() - packet->timestamp;
        total_latency += rtt;
        ping_count++;
        
        Serial.printf("Pong from " MACSTR " - RTT: %.2f ms (seq: %lu)\n", 
                     MAC2STR(addr()), rtt / 1000.0, packet->sequence);
        Serial.printf("  Avg RTT: %.2f ms, Sent: %lu, Received: %lu, Loss: %.1f%%\n",
                     (total_latency / ping_count) / 1000.0,
                     sequence_num,
                     ping_count,
                     ((sequence_num - ping_count) * 100.0) / sequence_num);
        
        waiting_for_pong = false;
      }
    }
  }

  bool send_ping() {
    if (waiting_for_pong && (micros() - last_ping_time > PING_TIMEOUT * 1000)) {
      Serial.println("Ping timeout!");
      waiting_for_pong = false;
    }

    if (!waiting_for_pong) {
      PingPacket packet;
      packet.type = PING;
      packet.sequence = sequence_num++;
      packet.timestamp = micros();
      
      if (send((uint8_t *)&packet, sizeof(packet))) {
        last_ping_time = packet.timestamp;
        waiting_for_pong = true;
        return true;
      } else {
        Serial.println("Failed to send ping");
        return false;
      }
    }
    return false;
  }

  void print_statistics() {
    Serial.println("\n--- Ping Statistics ---");
    Serial.printf("Packets sent: %lu\n", sequence_num);
    Serial.printf("Packets received: %lu\n", ping_count);
    Serial.printf("Packet loss: %.1f%%\n", ((sequence_num - ping_count) * 100.0) / sequence_num);
    if (ping_count > 0) {
      Serial.printf("Average RTT: %.2f ms\n", (total_latency / ping_count) / 1000.0);
    }
    Serial.println("-----------------------\n");
  }

private:
  enum PacketType { PING = 0, PONG = 1 };
  
  struct PingPacket {
    uint8_t type;
    uint32_t sequence;
    unsigned long timestamp;
  };

  unsigned long last_ping_time = 0;
  unsigned long total_latency = 0;
  uint32_t sequence_num = 0;
  uint32_t ping_count = 0;
  bool waiting_for_pong = false;
};

/* Global Variables */

// Replace with your slave's MAC address
uint8_t slave_mac[] = {0x38, 0x18, 0x2B, 0x8B, 0x87, 0xA4};  // Change this!

ESP_NOW_Ping_Peer *ping_peer = nullptr;

/* Main */

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== ESP-NOW Ping Test Master ===");
  Serial.println("IMPORTANT: Update slave_mac[] with your slave's MAC address!");

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

  // Initialize ESP-NOW
  if (!ESP_NOW.begin()) {
    Serial.println("Failed to initialize ESP-NOW");
    Serial.println("Rebooting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  // Create and register peer
  ping_peer = new ESP_NOW_Ping_Peer(slave_mac, ESPNOW_WIFI_CHANNEL, WIFI_IF_STA, nullptr);
  if (!ping_peer->begin()) {
    Serial.println("Failed to register peer");
    Serial.println("Rebooting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  Serial.println("\nSetup complete. Starting ping test...\n");
}

void loop() {
  static unsigned long last_ping = 0;
  static unsigned long last_stats = 0;

  // Send ping at regular intervals
  if (millis() - last_ping >= PING_INTERVAL) {
    last_ping = millis();
    ping_peer->send_ping();
  }

  // Print statistics every 10 seconds
  if (millis() - last_stats >= 10000) {
    last_stats = millis();
    ping_peer->print_statistics();
  }

  delay(10);
}