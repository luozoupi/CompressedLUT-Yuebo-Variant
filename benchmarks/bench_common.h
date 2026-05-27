#ifndef COMPRESSEDLUT_BENCH_COMMON_H
#define COMPRESSEDLUT_BENCH_COMMON_H

#include "../compressedlut.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace clut_bench {

enum FunctionKind {
    FunctionNone = 0,
    FunctionExpShift = 1,
    FunctionSigmoid = 2,
    FunctionSqrt = 3,
    FunctionSinHalfPi = 4,
    FunctionLog1p = 5,
    FunctionTanh = 6
};

struct Dataset {
    std::string name;
    std::vector<long int> table;
    FunctionKind function_kind = FunctionNone;
    int f_in = 0;
    int f_out = 0;
};

struct CompressionResult {
    compressedlut::CompressedTable artifact;
    double build_ms = 0.0;
};

inline double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

inline double median(std::vector<double> values)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if(values.size() % 2 == 1)
        return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

inline uint64_t mix_checksum(uint64_t checksum, uint64_t value)
{
    return checksum + ((value + 0x9e3779b97f4a7c15ULL) * 0xbf58476d1ce4e5b9ULL);
}

inline double eval_function(FunctionKind kind, double x)
{
    const double pi = 3.141592653589793238462643383279502884;
    switch(kind)
    {
        case FunctionExpShift:
            return std::exp(x - 1.0);
        case FunctionSigmoid:
            return 1.0 / (1.0 + std::exp(-8.0 * (x - 0.5)));
        case FunctionSqrt:
            return std::sqrt(x);
        case FunctionSinHalfPi:
            return std::sin(0.5 * pi * x);
        case FunctionLog1p:
            return std::log1p(x);
        case FunctionTanh:
            return std::tanh(3.0 * x);
        case FunctionNone:
        default:
            return 0.0;
    }
}

inline long int quantize_function(FunctionKind kind, double x, int f_out)
{
    return static_cast<long int>(std::llround(eval_function(kind, x) * static_cast<double>(1L << f_out)));
}

inline std::vector<long int> generate_function_table(FunctionKind kind, int f_in, int f_out)
{
    const size_t entries = static_cast<size_t>(1) << f_in;
    std::vector<long int> table;
    table.reserve(entries);
    for(size_t i = 0; i < entries; i++)
    {
        const double x = static_cast<double>(i) / static_cast<double>(entries);
        table.push_back(quantize_function(kind, x, f_out));
    }
    return table;
}

inline std::vector<long int> read_hex_table(const std::string& path)
{
    std::ifstream file(path.c_str());
    if(!file)
        throw std::runtime_error("could not open table file: " + path);

    std::vector<long int> table;
    std::string line;
    while(std::getline(file, line))
    {
        if(line.empty())
            continue;
        table.push_back(std::stol(line, 0, 16));
    }
    return table;
}

inline void validate_power_of_two(size_t value, const std::string& name)
{
    if(value == 0 || (value & (value - 1)) != 0)
        throw std::runtime_error(name + " entry count is not a power of two");
}

inline std::vector<Dataset> make_datasets(const std::string& repo_root, bool quick)
{
    const int f_in = quick ? 10 : 12;
    const int f_out = 12;
    std::vector<Dataset> datasets;

    Dataset example;
    example.name = "example_txt";
    example.table = read_hex_table(repo_root + "/example.txt");
    validate_power_of_two(example.table.size(), example.name);
    datasets.push_back(example);

    const FunctionKind function_kinds[] = {
        FunctionExpShift,
        FunctionSigmoid,
        FunctionSqrt,
        FunctionSinHalfPi,
        FunctionLog1p,
        FunctionTanh
    };
    const char* function_names[] = {
        "exp_x_minus_1",
        "sigmoid_8x",
        "sqrt_x",
        "sin_half_pi_x",
        "log1p_x",
        "tanh_3x"
    };
    const size_t function_count = quick ? 2 : (sizeof(function_kinds) / sizeof(function_kinds[0]));

    for(size_t i = 0; i < function_count; i++)
    {
        Dataset dataset;
        dataset.name = function_names[i];
        dataset.function_kind = function_kinds[i];
        dataset.f_in = f_in;
        dataset.f_out = f_out;
        dataset.table = generate_function_table(dataset.function_kind, f_in, f_out);
        validate_power_of_two(dataset.table.size(), dataset.name);
        datasets.push_back(dataset);
    }

    return datasets;
}

inline CompressionResult compress_dataset(const Dataset& dataset)
{
    compressedlut::struct_configs configs = {2, 1, 1, 1};
    const double start = now_ms();
    compressedlut::CompressedTable artifact = compressedlut::compress_table_artifact(dataset.table, configs);
    const double end = now_ms();

    CompressionResult result;
    result.artifact = artifact;
    result.build_ms = end - start;
    return result;
}

inline bool validate_decode(const Dataset& dataset, const compressedlut::CompressedTable& artifact, std::string* error)
{
    if(!artifact.is_compressed())
    {
        if(error)
            *error = "table did not compress";
        return false;
    }

    for(size_t i = 0; i < dataset.table.size(); i++)
    {
        const long int decoded = compressedlut::decode(artifact, static_cast<long int>(i));
        if(decoded != dataset.table[i])
        {
            if(error)
            {
                std::ostringstream os;
                os << "decode mismatch at " << i << ": decoded=" << decoded << " expected=" << dataset.table[i];
                *error = os.str();
            }
            return false;
        }
    }
    return true;
}

inline std::vector<uint32_t> make_addresses(size_t entries, size_t lookups, const std::string& pattern)
{
    validate_power_of_two(entries, "address table");
    std::vector<uint32_t> addresses(lookups);
    const uint32_t mask = static_cast<uint32_t>(entries - 1);

    if(pattern == "sequential")
    {
        for(size_t i = 0; i < lookups; i++)
            addresses[i] = static_cast<uint32_t>(i) & mask;
    }
    else if(pattern == "random")
    {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<uint32_t> dist(0, mask);
        for(size_t i = 0; i < lookups; i++)
            addresses[i] = dist(rng);
    }
    else
    {
        throw std::runtime_error("unknown address pattern: " + pattern);
    }

    return addresses;
}

inline std::vector<int> parse_int_list(const std::string& value)
{
    std::vector<int> result;
    std::stringstream ss(value);
    std::string item;
    while(std::getline(ss, item, ','))
    {
        if(!item.empty())
            result.push_back(std::stoi(item));
    }
    return result;
}

inline long int final_compressed_bits(const compressedlut::CompressedTable& artifact)
{
    if(artifact.final_size.empty())
        return 0;
    return artifact.final_size.back();
}

inline int output_bits(const Dataset& dataset)
{
    if(dataset.table.empty())
        return 0;
    return compressedlut::bit_width(*std::max_element(dataset.table.begin(), dataset.table.end()));
}

inline void write_csv_header(std::ostream& os)
{
    os << "dataset,backend,variant,address_pattern,threads,lookups,median_ms,mops,ns_per_lookup,checksum,"
       << "initial_bits,compressed_bits,compression_ratio,levels,entries,output_bits,compression_ms,status\n";
}

inline void write_csv_row(
    std::ostream& os,
    const std::string& dataset,
    const std::string& backend,
    const std::string& variant,
    const std::string& pattern,
    int threads,
    size_t lookups,
    double median_ms,
    uint64_t checksum,
    long int initial_bits,
    long int compressed_bits,
    size_t levels,
    size_t entries,
    int out_bits,
    double compression_ms,
    const std::string& status)
{
    const double mops = median_ms > 0.0 ? (static_cast<double>(lookups) / median_ms / 1000.0) : 0.0;
    const double ns_per_lookup = lookups > 0 ? (median_ms * 1000000.0 / static_cast<double>(lookups)) : 0.0;
    const double ratio = initial_bits > 0 ? (static_cast<double>(compressed_bits) / static_cast<double>(initial_bits)) : 0.0;

    os << dataset << ','
       << backend << ','
       << variant << ','
       << pattern << ','
       << threads << ','
       << lookups << ','
       << std::fixed << std::setprecision(6) << median_ms << ','
       << std::fixed << std::setprecision(6) << mops << ','
       << std::fixed << std::setprecision(6) << ns_per_lookup << ','
       << checksum << ','
       << initial_bits << ','
       << compressed_bits << ','
       << std::fixed << std::setprecision(6) << ratio << ','
       << levels << ','
       << entries << ','
       << out_bits << ','
       << std::fixed << std::setprecision(6) << compression_ms << ','
       << status << '\n';
}

} // namespace clut_bench

#endif
