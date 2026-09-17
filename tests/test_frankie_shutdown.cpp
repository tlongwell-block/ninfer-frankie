#include "cpp-httplib/httplib.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

// A peer that never reads frames cannot complete a WebSocket close handshake.
// Explicit shutdown must still release the blocked server read and listener.
static void check_shutdown(bool microphone) {
    using namespace std::chrono_literals;
    httplib::Server server;
    std::mutex mutex;
    std::condition_variable ready;
    httplib::ws::WebSocket* active = nullptr;
    std::atomic<bool> finished{false}, stop_microphone{false};
    server.WebSocket("/v1/realtime", [&](const auto&, auto& socket) {
        {
            std::lock_guard lock(mutex);
            active = &socket;
            ready.notify_all();
        }
        std::string message;
        while (socket.read(message) == httplib::ws::Text) {}
        std::lock_guard lock(mutex);
        active = nullptr;
    });
    const auto port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) { throw std::runtime_error("bind failed"); }
    std::thread listener([&] { server.listen_after_bind(); finished = true; });
    httplib::ws::WebSocketClient client("ws://127.0.0.1:" + std::to_string(port) + "/v1/realtime");
    const auto connected = client.connect();
    bool admitted = false;
    if (connected) {
        std::unique_lock lock(mutex);
        admitted = ready.wait_for(lock, 2s, [&] { return active != nullptr; });
    }
    std::thread sender;
    if (admitted && microphone) {
        sender = std::thread([&] {
            while (!stop_microphone && client.send("{\"type\":\"input_audio_buffer.append\",\"audio\":\"AAAA\"}")) {
                std::this_thread::sleep_for(5ms);
            }
        });
    }
    std::this_thread::sleep_for(20ms);
    const auto started = std::chrono::steady_clock::now();
    server.stop();
    {
        std::lock_guard lock(mutex);
        if (active) { active->shutdown(); }
    }
    while (!finished && std::chrono::steady_clock::now() - started < 2s) {
        std::this_thread::sleep_for(5ms);
    }
    const bool stopped_without_peer = finished;
    stop_microphone = true;
    if (sender.joinable()) { sender.join(); }
    client.close();
    listener.join();
    if (!admitted || !stopped_without_peer) {
        throw std::runtime_error("shutdown waited for an unresponsive realtime peer");
    }
}

int main() {
    check_shutdown(false);
    check_shutdown(true);
    std::cout << "PASS: realtime shutdown interrupts idle and active peers without a close reply\n";
}
