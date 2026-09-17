#pragma once

#include "runtime-options.h"
#include "serve/serve_options.h"

namespace ninfer::frankie {

struct Options {
    serve::ServeOptions http;
    frankie_options voice;
    std::string package;
};

Options parse_options(int argc, char** argv);
std::string usage(const char* program);

} // namespace ninfer::frankie
