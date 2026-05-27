#include "bench_common.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace {

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if(err__ != cudaSuccess) { \
        std::ostringstream os__; \
        os__ << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": " << cudaGetErrorString(err__); \
        throw std::runtime_error(os__.str()); \
    } \
} while(0)

struct Options {
    std::string repo_root = ".";
    std::string out_path;
    size_t lookups = static_cast<size_t>(1) << 24;
    int repeats = 5;
    int device = 7;
    int block_size = 256;
    bool quick = false;
    std::string dataset_filter;
    std::string variant_filter;
    std::string pattern_filter;
};

struct TimedResult {
    double median_ms = 0.0;
    uint64_t checksum = 0;
};

struct DeviceLevel {
    int w_in;
    int w_out;
    int w_l;
    int w_s;
    int w_lb;
    int w_ust;
    int w_bias;
    int w_idx;
    int w_rsh;
    int has_idx;
    const uint32_t* lb;
    const uint32_t* ust;
    const uint32_t* bias;
    const uint32_t* idx;
    const uint32_t* rsh;
};

struct DeviceCompressedTable {
    DeviceLevel* levels = NULL;
    int num_levels = 0;
    std::vector<uint32_t*> allocations;

    ~DeviceCompressedTable()
    {
        release();
    }

    void release()
    {
        for(size_t i = 0; i < allocations.size(); i++)
        {
            if(allocations[i] != NULL)
                cudaFree(allocations[i]);
        }
        allocations.clear();
        if(levels != NULL)
        {
            cudaFree(levels);
            levels = NULL;
        }
        num_levels = 0;
    }

    uint32_t* copy_vector(const std::vector<long int>& values)
    {
        if(values.empty())
            return NULL;
        std::vector<uint32_t> packed(values.size());
        for(size_t i = 0; i < values.size(); i++)
            packed[i] = static_cast<uint32_t>(values[i]);

        uint32_t* device_ptr = NULL;
        CUDA_CHECK(cudaMalloc(&device_ptr, packed.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(device_ptr, packed.data(), packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        allocations.push_back(device_ptr);
        return device_ptr;
    }

    void upload(const compressedlut::CompressedTable& artifact)
    {
        release();
        num_levels = static_cast<int>(artifact.levels.size());
        std::vector<DeviceLevel> host_levels(static_cast<size_t>(num_levels));

        for(int i = 0; i < num_levels; i++)
        {
            const compressedlut::CompressedLevel& level = artifact.levels[static_cast<size_t>(i)];
            DeviceLevel desc;
            std::memset(&desc, 0, sizeof(desc));
            desc.w_in = level.w_in;
            desc.w_out = level.w_out;
            desc.w_l = level.w_l;
            desc.w_s = level.w_s;
            desc.w_lb = level.w_lb;
            desc.w_ust = level.w_ust;
            desc.w_bias = level.w_bias;
            desc.w_idx = level.w_idx;
            desc.w_rsh = level.w_rsh;
            desc.has_idx = level.t_idx.empty() ? 0 : 1;
            desc.lb = copy_vector(level.t_lb);
            desc.ust = copy_vector(level.t_ust);
            desc.bias = copy_vector(level.t_bias);
            desc.idx = copy_vector(level.t_idx);
            desc.rsh = copy_vector(level.t_rsh);
            host_levels[static_cast<size_t>(i)] = desc;
        }

        CUDA_CHECK(cudaMalloc(&levels, host_levels.size() * sizeof(DeviceLevel)));
        CUDA_CHECK(cudaMemcpy(levels, host_levels.data(), host_levels.size() * sizeof(DeviceLevel), cudaMemcpyHostToDevice));
    }
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for(int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if(arg == "--repo-root" && i + 1 < argc)
            options.repo_root = argv[++i];
        else if(arg == "--out" && i + 1 < argc)
            options.out_path = argv[++i];
        else if(arg == "--lookups" && i + 1 < argc)
            options.lookups = static_cast<size_t>(std::stoull(argv[++i]));
        else if(arg == "--repeats" && i + 1 < argc)
            options.repeats = std::stoi(argv[++i]);
        else if(arg == "--device" && i + 1 < argc)
            options.device = std::stoi(argv[++i]);
        else if(arg == "--block-size" && i + 1 < argc)
            options.block_size = std::stoi(argv[++i]);
        else if(arg == "--dataset" && i + 1 < argc)
            options.dataset_filter = argv[++i];
        else if(arg == "--variant" && i + 1 < argc)
            options.variant_filter = argv[++i];
        else if(arg == "--pattern" && i + 1 < argc)
            options.pattern_filter = argv[++i];
        else if(arg == "--quick")
        {
            options.quick = true;
            options.lookups = static_cast<size_t>(1) << 20;
            options.repeats = 2;
        }
        else
            throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if(options.repeats < 1)
        options.repeats = 1;
    if(options.block_size < 32)
        options.block_size = 32;
    return options;
}

bool selected(const std::string& filter, const std::string& value)
{
    return filter.empty() || filter == value;
}

uint32_t* copy_u32_to_device(const std::vector<uint32_t>& values)
{
    uint32_t* ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, values.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(ptr, values.data(), values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return ptr;
}

std::vector<uint32_t> pack_table(const std::vector<long int>& table)
{
    std::vector<uint32_t> packed(table.size());
    for(size_t i = 0; i < table.size(); i++)
        packed[i] = static_cast<uint32_t>(table[i]);
    return packed;
}

uint64_t checksum_device_output(uint32_t* d_out, size_t count)
{
    std::vector<uint32_t> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), d_out, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint64_t checksum = 0;
    for(size_t i = 0; i < host.size(); i++)
        checksum = clut_bench::mix_checksum(checksum, host[i]);
    return checksum;
}

template <typename Launch>
TimedResult measure_cuda(size_t lookups, int repeats, uint32_t* d_out, const Launch& launch)
{
    TimedResult result;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> times;
    times.reserve(static_cast<size_t>(repeats));
    for(int r = 0; r < repeats; r++)
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        times.push_back(static_cast<double>(elapsed_ms));
    }

    result.checksum = checksum_device_output(d_out, lookups);
    result.median_ms = clut_bench::median(times);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return result;
}

void emit_row(
    std::ostream& os,
    const clut_bench::Dataset& dataset,
    const clut_bench::CompressionResult& compression,
    const std::string& variant,
    const std::string& pattern,
    int block_size,
    size_t lookups,
    const TimedResult& result,
    const std::string& status)
{
    const bool compressed = compression.artifact.is_compressed();
    const int out_bits = clut_bench::output_bits(dataset);
    const long int initial_bits = compressed ? compression.artifact.initial_size : static_cast<long int>(dataset.table.size()) * out_bits;
    const long int compressed_bits = compressed ? clut_bench::final_compressed_bits(compression.artifact) : initial_bits;

    clut_bench::write_csv_row(
        os,
        dataset.name,
        "cuda",
        variant,
        pattern,
        block_size,
        lookups,
        result.median_ms,
        result.checksum,
        initial_bits,
        compressed_bits,
        compression.artifact.levels.size(),
        dataset.table.size(),
        out_bits,
        compression.build_ms,
        status);
}

} // namespace

__device__ __forceinline__ uint32_t load_ro(const uint32_t* ptr, uint32_t index)
{
    return __ldg(ptr + index);
}

__device__ uint32_t decode_compresslut_device(const DeviceLevel* levels, int num_levels, uint32_t address)
{
    uint32_t addresses[16];
    addresses[0] = address;
    for(int i = 1; i < num_levels && i < 16; i++)
        addresses[i] = addresses[i - 1] >> levels[i - 1].w_s;

    uint32_t decoded = 0;
    for(int i = num_levels - 1; i >= 0; i--)
    {
        const DeviceLevel level = levels[i];
        const uint32_t level_address = addresses[i];
        const uint32_t high_address = level_address >> level.w_s;
        const uint32_t low_mask = (1u << level.w_s) - 1u;
        const uint32_t low_address = level_address & low_mask;

        uint32_t bias = 0;
        if(level.w_bias != 0)
            bias = (i + 1 < num_levels) ? decoded : load_ro(level.bias, high_address);

        uint32_t ust_value = 0;
        if(level.w_ust != 0)
        {
            uint32_t ust_address = level_address;
            if(level.has_idx)
            {
                if(level.w_idx != 0)
                    ust_address = (load_ro(level.idx, high_address) << level.w_s) | low_address;
                else
                    ust_address = low_address;
            }
            ust_value = load_ro(level.ust, ust_address);
        }

        const uint32_t shift = (level.w_rsh != 0) ? load_ro(level.rsh, high_address) : 0u;
        const uint32_t high_value = ((level.w_ust != 0) ? (ust_value >> shift) : 0u) + ((level.w_bias != 0) ? bias : 0u);

        if(level.w_l != 0)
        {
            const uint32_t lb = (level.w_lb != 0) ? load_ro(level.lb, level_address) : 0u;
            decoded = (high_value << level.w_l) | lb;
        }
        else
        {
            decoded = high_value;
        }
    }
    return decoded;
}

__device__ double eval_function_device(int kind, double x)
{
    const double pi = 3.141592653589793238462643383279502884;
    switch(kind)
    {
        case clut_bench::FunctionExpShift:
            return exp(x - 1.0);
        case clut_bench::FunctionSigmoid:
            return 1.0 / (1.0 + exp(-8.0 * (x - 0.5)));
        case clut_bench::FunctionSqrt:
            return sqrt(x);
        case clut_bench::FunctionSinHalfPi:
            return sin(0.5 * pi * x);
        case clut_bench::FunctionLog1p:
            return log1p(x);
        case clut_bench::FunctionTanh:
            return tanh(3.0 * x);
        default:
            return 0.0;
    }
}

__global__ void plain_lut_kernel(const uint32_t* table, const uint32_t* addresses, uint32_t* out, unsigned long long count)
{
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    for(; i < count; i += stride)
        out[i] = load_ro(table, addresses[i]);
}

__global__ void compresslut_kernel(const DeviceLevel* levels, int num_levels, const uint32_t* addresses, uint32_t* out, unsigned long long count)
{
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    for(; i < count; i += stride)
        out[i] = decode_compresslut_device(levels, num_levels, addresses[i]);
}

__global__ void math_kernel(int function_kind, int f_out, unsigned long long entries, const uint32_t* addresses, uint32_t* out, unsigned long long count)
{
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    const double scale = static_cast<double>(1ULL << f_out);
    const double inv_entries = 1.0 / static_cast<double>(entries);
    for(; i < count; i += stride)
    {
        const double x = static_cast<double>(addresses[i]) * inv_entries;
        out[i] = static_cast<uint32_t>(llround(eval_function_device(function_kind, x) * scale));
    }
}

bool validate_cuda_decode(
    const clut_bench::Dataset& dataset,
    const DeviceCompressedTable& device_table,
    int block_size,
    std::string* error)
{
    const size_t entries = dataset.table.size();
    std::vector<uint32_t> addresses(entries);
    for(size_t i = 0; i < entries; i++)
        addresses[i] = static_cast<uint32_t>(i);

    uint32_t* d_addresses = NULL;
    uint32_t* d_out = NULL;
    try
    {
        d_addresses = copy_u32_to_device(addresses);
        CUDA_CHECK(cudaMalloc(&d_out, entries * sizeof(uint32_t)));
        const int grid = static_cast<int>((entries + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size));
        compresslut_kernel<<<grid, block_size>>>(device_table.levels, device_table.num_levels, d_addresses, d_out, entries);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uint32_t> out(entries);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, entries * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for(size_t i = 0; i < entries; i++)
        {
            if(out[i] != static_cast<uint32_t>(dataset.table[i]))
            {
                if(error)
                {
                    std::ostringstream os;
                    os << "cuda decode mismatch at " << i << ": decoded=" << out[i] << " expected=" << dataset.table[i];
                    *error = os.str();
                }
                cudaFree(d_addresses);
                cudaFree(d_out);
                return false;
            }
        }
    }
    catch(...)
    {
        if(d_addresses != NULL)
            cudaFree(d_addresses);
        if(d_out != NULL)
            cudaFree(d_out);
        throw;
    }

    cudaFree(d_addresses);
    cudaFree(d_out);
    return true;
}

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if(options.device < 0 || options.device >= device_count)
        {
            std::ostringstream os;
            os << "requested CUDA device " << options.device << " but visible device count is " << device_count;
            throw std::runtime_error(os.str());
        }
        CUDA_CHECK(cudaSetDevice(options.device));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
        std::cerr << "cuda: device " << options.device << " " << prop.name << " sm_" << prop.major << prop.minor << "\n";

        std::vector<clut_bench::Dataset> datasets = clut_bench::make_datasets(options.repo_root, options.quick);

        std::ofstream file;
        std::ostream* out = &std::cout;
        if(!options.out_path.empty())
        {
            file.open(options.out_path.c_str());
            if(!file)
                throw std::runtime_error("could not open output file: " + options.out_path);
            out = &file;
        }
        clut_bench::write_csv_header(*out);

        const std::string patterns[] = {"sequential", "random"};
        for(size_t d = 0; d < datasets.size(); d++)
        {
            const clut_bench::Dataset& dataset = datasets[d];
            if(!selected(options.dataset_filter, dataset.name))
                continue;

            std::cerr << "cuda: compressing " << dataset.name << " (" << dataset.table.size() << " entries)\n";
            const clut_bench::CompressionResult compression = clut_bench::compress_dataset(dataset);

            std::vector<uint32_t> packed_table = pack_table(dataset.table);
            uint32_t* d_table = copy_u32_to_device(packed_table);
            uint32_t* d_out = NULL;
            CUDA_CHECK(cudaMalloc(&d_out, options.lookups * sizeof(uint32_t)));

            DeviceCompressedTable device_table;
            bool decode_ok = false;
            std::string validation_error;
            const bool needs_compresslut = selected(options.variant_filter, "compresslut");
            if(compression.artifact.is_compressed())
            {
                device_table.upload(compression.artifact);
                if(needs_compresslut)
                {
                    decode_ok = validate_cuda_decode(dataset, device_table, options.block_size, &validation_error);
                    if(!decode_ok)
                        std::cerr << "cuda: " << dataset.name << " decode validation failed: " << validation_error << "\n";
                }
                else
                {
                    decode_ok = true;
                }
            }
            else
            {
                validation_error = "table did not compress";
            }

            for(size_t p = 0; p < 2; p++)
            {
                const std::string pattern = patterns[p];
                if(!selected(options.pattern_filter, pattern))
                    continue;

                const std::vector<uint32_t> addresses = clut_bench::make_addresses(dataset.table.size(), options.lookups, pattern);
                uint32_t* d_addresses = copy_u32_to_device(addresses);
                const int grid = static_cast<int>((options.lookups + static_cast<size_t>(options.block_size) - 1) / static_cast<size_t>(options.block_size));

                if(selected(options.variant_filter, "plain_lut"))
                {
                    const TimedResult plain = measure_cuda(options.lookups, options.repeats, d_out, [&]() {
                        plain_lut_kernel<<<grid, options.block_size>>>(d_table, d_addresses, d_out, options.lookups);
                        CUDA_CHECK(cudaGetLastError());
                    });
                    emit_row(*out, dataset, compression, "plain_lut", pattern, options.block_size, options.lookups, plain, "ok");
                }

                if(selected(options.variant_filter, "compresslut") && decode_ok)
                {
                    const TimedResult clut = measure_cuda(options.lookups, options.repeats, d_out, [&]() {
                        compresslut_kernel<<<grid, options.block_size>>>(device_table.levels, device_table.num_levels, d_addresses, d_out, options.lookups);
                        CUDA_CHECK(cudaGetLastError());
                    });
                    emit_row(*out, dataset, compression, "compresslut", pattern, options.block_size, options.lookups, clut, "ok");
                }
                else if(selected(options.variant_filter, "compresslut"))
                {
                    TimedResult skipped;
                    emit_row(*out, dataset, compression, "compresslut", pattern, options.block_size, options.lookups, skipped, validation_error);
                }

                if(selected(options.variant_filter, "cuda_math_f64") && dataset.function_kind != clut_bench::FunctionNone)
                {
                    const TimedResult math = measure_cuda(options.lookups, options.repeats, d_out, [&]() {
                        math_kernel<<<grid, options.block_size>>>(
                            static_cast<int>(dataset.function_kind),
                            dataset.f_out,
                            dataset.table.size(),
                            d_addresses,
                            d_out,
                            options.lookups);
                        CUDA_CHECK(cudaGetLastError());
                    });
                    emit_row(*out, dataset, compression, "cuda_math_f64", pattern, options.block_size, options.lookups, math, "ok");
                }

                cudaFree(d_addresses);
            }

            cudaFree(d_out);
            cudaFree(d_table);
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "bench_cuda error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
