#include "../bench_common.h"

#include <cuda_runtime.h>

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
    std::string dataset_filter;
    std::string pattern_filter;
    size_t lookups = static_cast<size_t>(1) << 24;
    int repeats = 5;
    int device = 7;
    int block_size = 256;
    bool quick = false;
};

struct TimedResult {
    double median_ms = 0.0;
    uint64_t checksum = 0;
};

struct DeviceV2Table {
    int w_l = 0;
    int w_s = 0;
    int w_lb = 0;
    int w_ust = 0;
    int w_bias = 0;
    int w_idx = 0;
    int w_rsh = 0;
    int has_idx = 0;
    const uint16_t* lb = NULL;
    const uint16_t* ust = NULL;
    const uint16_t* bias = NULL;
    const uint16_t* idx = NULL;
    const uint16_t* rsh = NULL;
};

struct HostV2Table {
    DeviceV2Table desc;
    DeviceV2Table* d_desc = NULL;
    std::vector<uint16_t*> allocations;
    long int storage_bits = 0;
    long int runtime_bits = 0;
    size_t runtime_bytes = 0;
    size_t selected_levels = 1;

    ~HostV2Table()
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
        if(d_desc != NULL)
        {
            cudaFree(d_desc);
            d_desc = NULL;
        }
    }

    uint16_t* copy_vector(const std::vector<long int>& values, const std::string& name)
    {
        if(values.empty())
            return NULL;

        std::vector<uint16_t> packed(values.size());
        for(size_t i = 0; i < values.size(); i++)
        {
            if(values[i] < 0 || values[i] > 65535)
                throw std::runtime_error("v2 u16 runtime cannot pack " + name);
            packed[i] = static_cast<uint16_t>(values[i]);
        }

        uint16_t* device_ptr = NULL;
        CUDA_CHECK(cudaMalloc(&device_ptr, packed.size() * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(device_ptr, packed.data(), packed.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        allocations.push_back(device_ptr);
        runtime_bytes += packed.size() * sizeof(uint16_t);
        return device_ptr;
    }

    void upload(const compressedlut::CompressedTable& artifact)
    {
        release();
        if(artifact.levels.empty())
            throw std::runtime_error("cannot build v2 table from uncompressed artifact");

        const compressedlut::CompressedLevel& level = artifact.levels.front();
        std::memset(&desc, 0, sizeof(desc));
        desc.w_l = level.w_l;
        desc.w_s = level.w_s;
        desc.w_lb = level.w_lb;
        desc.w_ust = level.w_ust;
        desc.w_bias = level.w_bias;
        desc.w_idx = level.w_idx;
        desc.w_rsh = level.w_rsh;
        desc.has_idx = level.t_idx.empty() ? 0 : 1;

        desc.lb = copy_vector(level.t_lb, "lb");
        desc.ust = copy_vector(level.t_ust, "ust");
        desc.bias = copy_vector(level.t_bias, "bias");
        desc.idx = copy_vector(level.t_idx, "idx");
        desc.rsh = copy_vector(level.t_rsh, "rsh");

        storage_bits = clut_bench::final_compressed_bits(artifact);
        runtime_bits = artifact.final_size.empty() ? 0 : artifact.final_size.front();
        selected_levels = 1;

        CUDA_CHECK(cudaMalloc(&d_desc, sizeof(DeviceV2Table)));
        CUDA_CHECK(cudaMemcpy(d_desc, &desc, sizeof(DeviceV2Table), cudaMemcpyHostToDevice));
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

void write_v2_header(std::ostream& os)
{
    os << "dataset,backend,variant,address_pattern,threads,lookups,median_ms,mops,ns_per_lookup,checksum,"
       << "initial_bits,storage_compressed_bits,runtime_bits,runtime_bytes,storage_ratio,runtime_ratio,"
       << "levels,selected_levels,entries,output_bits,compression_ms,status\n";
}

void write_v2_row(
    std::ostream& os,
    const clut_bench::Dataset& dataset,
    const clut_bench::CompressionResult& compression,
    const HostV2Table& table,
    const std::string& pattern,
    int block_size,
    size_t lookups,
    const TimedResult& result,
    const std::string& status)
{
    const int out_bits = clut_bench::output_bits(dataset);
    const long int initial_bits = compression.artifact.initial_size;
    const double mops = result.median_ms > 0.0 ? (static_cast<double>(lookups) / result.median_ms / 1000.0) : 0.0;
    const double ns_per_lookup = lookups > 0 ? (result.median_ms * 1000000.0 / static_cast<double>(lookups)) : 0.0;
    const double storage_ratio = initial_bits > 0 ? static_cast<double>(table.storage_bits) / static_cast<double>(initial_bits) : 0.0;
    const double runtime_ratio = initial_bits > 0 ? static_cast<double>(table.runtime_bits) / static_cast<double>(initial_bits) : 0.0;

    os << dataset.name << ','
       << "cuda_v2" << ','
       << "expand_bias_u16" << ','
       << pattern << ','
       << block_size << ','
       << lookups << ','
       << std::fixed << std::setprecision(6) << result.median_ms << ','
       << std::fixed << std::setprecision(6) << mops << ','
       << std::fixed << std::setprecision(6) << ns_per_lookup << ','
       << result.checksum << ','
       << initial_bits << ','
       << table.storage_bits << ','
       << table.runtime_bits << ','
       << table.runtime_bytes << ','
       << std::fixed << std::setprecision(6) << storage_ratio << ','
       << std::fixed << std::setprecision(6) << runtime_ratio << ','
       << compression.artifact.levels.size() << ','
       << table.selected_levels << ','
       << dataset.table.size() << ','
       << out_bits << ','
       << std::fixed << std::setprecision(6) << compression.build_ms << ','
       << status << '\n';
}

} // namespace

__device__ __forceinline__ uint32_t load_u16_ro(const uint16_t* ptr, uint32_t index)
{
    return static_cast<uint32_t>(__ldg(ptr + index));
}

__global__ void compresslut_v2_expand_bias_kernel(const DeviceV2Table* table, const uint32_t* addresses, uint32_t* out, unsigned long long count)
{
    const DeviceV2Table t = *table;
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    const uint32_t low_mask = (1u << t.w_s) - 1u;

    for(; i < count; i += stride)
    {
        const uint32_t address = addresses[i];
        const uint32_t high_address = address >> t.w_s;
        const uint32_t low_address = address & low_mask;

        uint32_t ust_address = address;
        if(t.has_idx)
        {
            if(t.w_idx != 0)
                ust_address = (load_u16_ro(t.idx, high_address) << t.w_s) | low_address;
            else
                ust_address = low_address;
        }

        const uint32_t shift = (t.w_rsh != 0) ? load_u16_ro(t.rsh, high_address) : 0u;
        const uint32_t u = (t.w_ust != 0) ? load_u16_ro(t.ust, ust_address) : 0u;
        const uint32_t bias = (t.w_bias != 0) ? load_u16_ro(t.bias, high_address) : 0u;
        const uint32_t high_value = ((t.w_ust != 0) ? (u >> shift) : 0u) + bias;

        if(t.w_l != 0)
        {
            const uint32_t lb = (t.w_lb != 0) ? load_u16_ro(t.lb, address) : 0u;
            out[i] = (high_value << t.w_l) | lb;
        }
        else
        {
            out[i] = high_value;
        }
    }
}

bool validate_v2_decode(const clut_bench::Dataset& dataset, const HostV2Table& table, int block_size, std::string* error)
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
        compresslut_v2_expand_bias_kernel<<<grid, block_size>>>(table.d_desc, d_addresses, d_out, entries);
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
                    os << "v2 decode mismatch at " << i << ": decoded=" << out[i] << " expected=" << dataset.table[i];
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
        std::cerr << "cuda_v2: device " << options.device << " " << prop.name << " sm_" << prop.major << prop.minor << "\n";

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
        write_v2_header(*out);

        const std::string patterns[] = {"sequential", "random"};
        for(size_t d = 0; d < datasets.size(); d++)
        {
            const clut_bench::Dataset& dataset = datasets[d];
            if(!selected(options.dataset_filter, dataset.name))
                continue;

            std::cerr << "cuda_v2: compressing " << dataset.name << " (" << dataset.table.size() << " entries)\n";
            const clut_bench::CompressionResult compression = clut_bench::compress_dataset(dataset);
            if(!compression.artifact.is_compressed())
            {
                std::cerr << "cuda_v2: skipping uncompressed dataset " << dataset.name << "\n";
                continue;
            }

            HostV2Table table;
            table.upload(compression.artifact);

            std::string validation_error;
            const bool decode_ok = validate_v2_decode(dataset, table, options.block_size, &validation_error);
            if(!decode_ok)
                std::cerr << "cuda_v2: " << dataset.name << " validation failed: " << validation_error << "\n";

            uint32_t* d_out = NULL;
            CUDA_CHECK(cudaMalloc(&d_out, options.lookups * sizeof(uint32_t)));

            for(size_t p = 0; p < 2; p++)
            {
                const std::string pattern = patterns[p];
                if(!selected(options.pattern_filter, pattern))
                    continue;

                const std::vector<uint32_t> addresses = clut_bench::make_addresses(dataset.table.size(), options.lookups, pattern);
                uint32_t* d_addresses = copy_u32_to_device(addresses);
                const int grid = static_cast<int>((options.lookups + static_cast<size_t>(options.block_size) - 1) / static_cast<size_t>(options.block_size));

                if(decode_ok)
                {
                    const TimedResult timed = measure_cuda(options.lookups, options.repeats, d_out, [&]() {
                        compresslut_v2_expand_bias_kernel<<<grid, options.block_size>>>(table.d_desc, d_addresses, d_out, options.lookups);
                        CUDA_CHECK(cudaGetLastError());
                    });
                    write_v2_row(*out, dataset, compression, table, pattern, options.block_size, options.lookups, timed, "ok");
                }
                else
                {
                    TimedResult skipped;
                    write_v2_row(*out, dataset, compression, table, pattern, options.block_size, options.lookups, skipped, validation_error);
                }

                cudaFree(d_addresses);
            }

            cudaFree(d_out);
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "bench_cuda_v2 error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
