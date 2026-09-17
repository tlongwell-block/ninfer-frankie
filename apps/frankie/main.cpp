#include "options.h"
#include "ninfer-brain.h"
#include "mouth-session.h"
#include "realtime-server.h"
#include "common.h"
#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"

#include <atomic>
#include <csignal>
#include <chrono>
#include <thread>
#include <iostream>

namespace {
// atomic_flag is always lock-free; socket shutdown runs on a normal thread.
std::atomic_flag stop_requested = ATOMIC_FLAG_INIT;
void stop_server(int) { stop_requested.test_and_set(std::memory_order_relaxed); }
}

int main(int argc, char** argv) {
    try {
        auto options = ninfer::frankie::parse_options(argc, argv);
        if (options.http.help_requested) {
            std::cout << ninfer::frankie::usage(argv[0]);
            return 0;
        }
        ninfer::product::LoggingRuntime logging({
            .logger_name = "ninfer-frankie", .level = options.http.log_level,
            .presentation = ninfer::product::LogPresentation::Service});
        ninfer::product::StartupLogRenderer startup_log(logging);
        ninfer::serve::OperationalLog operational_log(logging.logger());
        ninfer::serve::HttpServer server(options.http, logging.logger());
        if (!server.bind()) { throw std::runtime_error("could not bind HTTP listener"); }

        common_init();
        ninfer::serve::GenerationService service(options.http, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        options.voice.execute_device = [&](const std::function<void()>& work) {
            service.engine().with_device_idle(work);
        };
        ninfer_brain_session brain(service.engine(), options.package, options.voice);
        mouth_session mouth(options.package, options.voice, [&](const std::vector<float>& audio) {
            return frankie_reference_transcript(brain, audio);
        });
        mouth.warmup();
        mouth.report_memory();
        service.warmup();
        server.attach(service);
        frankie_realtime_routes realtime(server.transport(), brain, mouth, options.package, options.voice);

        std::signal(SIGINT, stop_server);
        std::signal(SIGTERM, stop_server);
        std::jthread shutdown([&](std::stop_token done) {
            while (!done.stop_requested() && !stop_requested.test(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (stop_requested.test(std::memory_order_relaxed)) {
                server.stop();
                realtime.stop();
            }
        });
        operational_log.engine_capacity(service);
        operational_log.server_ready(options.http.host, options.http.port, server.public_model_id(),
                                     !options.http.api_key.empty());
        const bool result = server.listen();
        if (!result && !stop_requested.test(std::memory_order_relaxed)) { throw std::runtime_error("HTTP listener failed"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer-frankie: " << error.what() << '\n';
        return 1;
    }
}
