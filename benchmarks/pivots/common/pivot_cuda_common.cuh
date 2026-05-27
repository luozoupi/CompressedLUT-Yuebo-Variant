#ifndef COMPRESSEDLUT_PIVOT_CUDA_COMMON_CUH
#define COMPRESSEDLUT_PIVOT_CUDA_COMMON_CUH

#include "../../bench_common.h"

#include <cuda_runtime.h>

#include <cstring>
#include <memory>

#define PIVOT_CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if(err__ != cudaSuccess) { \
        std::ostringstream os__; \
        os__ << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": " << cudaGetErrorString(err__); \
        throw std::runtime_error(os__.str()); \
    } \
} while(0)

namespace clut_pivot {

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

    HostV2Table() = default;
    HostV2Table(const HostV2Table&) = delete;
    HostV2Table& operator=(const HostV2Table&) = delete;

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
        runtime_bytes = 0;
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
        PIVOT_CUDA_CHECK(cudaMalloc(&device_ptr, packed.size() * sizeof(uint16_t)));
        PIVOT_CUDA_CHECK(cudaMemcpy(device_ptr, packed.data(), packed.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        allocations.push_back(device_ptr);
        runtime_bytes += packed.size() * sizeof(uint16_t);
        return device_ptr;
    }

    void upload(const compressedlut::CompressedTable& artifact, bool upload_descriptor = true)
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

        if(upload_descriptor)
        {
            PIVOT_CUDA_CHECK(cudaMalloc(&d_desc, sizeof(DeviceV2Table)));
            PIVOT_CUDA_CHECK(cudaMemcpy(d_desc, &desc, sizeof(DeviceV2Table), cudaMemcpyHostToDevice));
        }
    }
};

struct HostV2TableSet {
    std::vector<std::unique_ptr<HostV2Table>> tables;
    DeviceV2Table* d_descs = NULL;
    long int storage_bits = 0;
    long int runtime_bits = 0;
    size_t runtime_bytes = 0;

    ~HostV2TableSet()
    {
        release();
    }

    void release()
    {
        if(d_descs != NULL)
        {
            cudaFree(d_descs);
            d_descs = NULL;
        }
        tables.clear();
        storage_bits = 0;
        runtime_bits = 0;
        runtime_bytes = 0;
    }

    void upload(const std::vector<const compressedlut::CompressedTable*>& artifacts)
    {
        release();
        std::vector<DeviceV2Table> descs;
        descs.reserve(artifacts.size());
        tables.reserve(artifacts.size());
        for(size_t i = 0; i < artifacts.size(); i++)
        {
            std::unique_ptr<HostV2Table> table(new HostV2Table());
            table->upload(*artifacts[i], false);
            storage_bits += table->storage_bits;
            runtime_bits += table->runtime_bits;
            runtime_bytes += table->runtime_bytes;
            descs.push_back(table->desc);
            tables.push_back(std::move(table));
        }

        PIVOT_CUDA_CHECK(cudaMalloc(&d_descs, descs.size() * sizeof(DeviceV2Table)));
        PIVOT_CUDA_CHECK(cudaMemcpy(d_descs, descs.data(), descs.size() * sizeof(DeviceV2Table), cudaMemcpyHostToDevice));
    }
};

inline std::vector<uint32_t> make_group_ids(size_t groups, size_t count, const std::string& pattern)
{
    std::vector<uint32_t> ids(count);
    if(pattern == "sequential")
    {
        for(size_t i = 0; i < count; i++)
            ids[i] = static_cast<uint32_t>(i % groups);
    }
    else if(pattern == "random")
    {
        std::mt19937 rng(67890);
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(groups - 1));
        for(size_t i = 0; i < count; i++)
            ids[i] = dist(rng);
    }
    else
    {
        throw std::runtime_error("unknown group pattern: " + pattern);
    }
    return ids;
}

inline std::vector<uint32_t> make_codebook_ids(size_t groups, size_t unique_tables)
{
    std::vector<uint32_t> ids(groups);
    for(size_t i = 0; i < groups; i++)
        ids[i] = static_cast<uint32_t>(i % unique_tables);
    return ids;
}

inline uint32_t* copy_u32_to_device(const std::vector<uint32_t>& values)
{
    uint32_t* ptr = NULL;
    PIVOT_CUDA_CHECK(cudaMalloc(&ptr, values.size() * sizeof(uint32_t)));
    PIVOT_CUDA_CHECK(cudaMemcpy(ptr, values.data(), values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return ptr;
}

inline float* copy_f32_to_device(const std::vector<float>& values)
{
    float* ptr = NULL;
    PIVOT_CUDA_CHECK(cudaMalloc(&ptr, values.size() * sizeof(float)));
    PIVOT_CUDA_CHECK(cudaMemcpy(ptr, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    return ptr;
}

inline uint64_t checksum_u32_device(const uint32_t* d_out, size_t count)
{
    std::vector<uint32_t> host(count);
    PIVOT_CUDA_CHECK(cudaMemcpy(host.data(), d_out, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint64_t checksum = 0;
    for(size_t i = 0; i < host.size(); i++)
        checksum = clut_bench::mix_checksum(checksum, host[i]);
    return checksum;
}

inline uint64_t checksum_f32_device(const float* d_out, size_t count)
{
    std::vector<float> host(count);
    PIVOT_CUDA_CHECK(cudaMemcpy(host.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
    uint64_t checksum = 0;
    for(size_t i = 0; i < host.size(); i++)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &host[i], sizeof(bits));
        checksum = clut_bench::mix_checksum(checksum, bits);
    }
    return checksum;
}

template <typename Launch, typename Checksum>
TimedResult measure_cuda(size_t count, int repeats, const Launch& launch, const Checksum& checksum)
{
    TimedResult result;
    cudaEvent_t start;
    cudaEvent_t stop;
    PIVOT_CUDA_CHECK(cudaEventCreate(&start));
    PIVOT_CUDA_CHECK(cudaEventCreate(&stop));

    launch();
    PIVOT_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> times;
    times.reserve(static_cast<size_t>(repeats));
    for(int r = 0; r < repeats; r++)
    {
        PIVOT_CUDA_CHECK(cudaEventRecord(start));
        launch();
        PIVOT_CUDA_CHECK(cudaEventRecord(stop));
        PIVOT_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        PIVOT_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        times.push_back(static_cast<double>(elapsed_ms));
    }

    result.median_ms = clut_bench::median(times);
    result.checksum = checksum();

    PIVOT_CUDA_CHECK(cudaEventDestroy(start));
    PIVOT_CUDA_CHECK(cudaEventDestroy(stop));
    return result;
}

inline clut_bench::Dataset make_function_dataset(const std::string& name, clut_bench::FunctionKind kind, int f_in, int f_out)
{
    clut_bench::Dataset dataset;
    dataset.name = name;
    dataset.function_kind = kind;
    dataset.f_in = f_in;
    dataset.f_out = f_out;
    dataset.table = clut_bench::generate_function_table(kind, f_in, f_out);
    clut_bench::validate_power_of_two(dataset.table.size(), dataset.name);
    return dataset;
}

inline clut_bench::Dataset make_variant_dataset(const std::string& name, int f_in, int f_out, int variant)
{
    clut_bench::Dataset dataset = make_function_dataset(name, clut_bench::FunctionSigmoid, f_in, f_out);
    const size_t mask = dataset.table.size() - 1;
    std::vector<long int> shifted(dataset.table.size());
    const size_t shift = static_cast<size_t>((variant * 37) & static_cast<int>(mask));
    const long int bias = static_cast<long int>(variant & 31);
    for(size_t i = 0; i < dataset.table.size(); i++)
        shifted[i] = dataset.table[(i + shift) & mask] + bias;
    dataset.table.swap(shifted);
    return dataset;
}

inline void write_pivot_header(std::ostream& os)
{
    os << "direction,case_name,variant,address_pattern,entries,table_count,unique_tables,lookups,"
       << "median_ms,mops,ns_per_lookup,checksum,plain_bytes,compressed_storage_bits,runtime_bytes,"
       << "storage_ratio,runtime_ratio,compression_ms,status\n";
}

inline void write_pivot_row(
    std::ostream& os,
    const std::string& direction,
    const std::string& case_name,
    const std::string& variant,
    const std::string& pattern,
    size_t entries,
    size_t table_count,
    size_t unique_tables,
    size_t lookups,
    const TimedResult& result,
    size_t plain_bytes,
    long int compressed_storage_bits,
    size_t runtime_bytes,
    long int initial_bits,
    double compression_ms,
    const std::string& status)
{
    const double mops = result.median_ms > 0.0 ? (static_cast<double>(lookups) / result.median_ms / 1000.0) : 0.0;
    const double ns_per_lookup = lookups > 0 ? (result.median_ms * 1000000.0 / static_cast<double>(lookups)) : 0.0;
    const double storage_ratio = initial_bits > 0 ? static_cast<double>(compressed_storage_bits) / static_cast<double>(initial_bits) : 0.0;
    const double runtime_ratio = initial_bits > 0 ? static_cast<double>(runtime_bytes * 8) / static_cast<double>(initial_bits) : 0.0;

    os << direction << ','
       << case_name << ','
       << variant << ','
       << pattern << ','
       << entries << ','
       << table_count << ','
       << unique_tables << ','
       << lookups << ','
       << std::fixed << std::setprecision(6) << result.median_ms << ','
       << std::fixed << std::setprecision(6) << mops << ','
       << std::fixed << std::setprecision(6) << ns_per_lookup << ','
       << result.checksum << ','
       << plain_bytes << ','
       << compressed_storage_bits << ','
       << runtime_bytes << ','
       << std::fixed << std::setprecision(6) << storage_ratio << ','
       << std::fixed << std::setprecision(6) << runtime_ratio << ','
       << std::fixed << std::setprecision(6) << compression_ms << ','
       << status << '\n';
}

} // namespace clut_pivot

__device__ __forceinline__ uint32_t pivot_load_u16_ro(const uint16_t* ptr, uint32_t index)
{
    return static_cast<uint32_t>(__ldg(ptr + index));
}

__device__ __forceinline__ uint32_t pivot_decode_v2_value(const clut_pivot::DeviceV2Table& t, uint32_t address)
{
    const uint32_t low_mask = (1u << t.w_s) - 1u;
    const uint32_t high_address = address >> t.w_s;
    const uint32_t low_address = address & low_mask;

    uint32_t ust_address = address;
    if(t.has_idx)
    {
        if(t.w_idx != 0)
            ust_address = (pivot_load_u16_ro(t.idx, high_address) << t.w_s) | low_address;
        else
            ust_address = low_address;
    }

    const uint32_t shift = (t.w_rsh != 0) ? pivot_load_u16_ro(t.rsh, high_address) : 0u;
    const uint32_t u = (t.w_ust != 0) ? pivot_load_u16_ro(t.ust, ust_address) : 0u;
    const uint32_t bias = (t.w_bias != 0) ? pivot_load_u16_ro(t.bias, high_address) : 0u;
    const uint32_t high_value = ((t.w_ust != 0) ? (u >> shift) : 0u) + bias;

    if(t.w_l == 0)
        return high_value;

    const uint32_t lb = (t.w_lb != 0) ? pivot_load_u16_ro(t.lb, address) : 0u;
    return (high_value << t.w_l) | lb;
}

__global__ void pivot_plain_lut_kernel(const uint32_t* table, const uint32_t* addresses, uint32_t* out, unsigned long long count)
{
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    for(; i < count; i += stride)
        out[i] = __ldg(table + addresses[i]);
}

__global__ void pivot_v2_lut_kernel(const clut_pivot::DeviceV2Table* table, const uint32_t* addresses, uint32_t* out, unsigned long long count)
{
    const clut_pivot::DeviceV2Table t = *table;
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    for(; i < count; i += stride)
        out[i] = pivot_decode_v2_value(t, addresses[i]);
}

__global__ void pivot_plain_many_lut_kernel(
    const uint32_t* tables,
    uint32_t entries,
    const uint32_t* table_ids,
    const uint32_t* addresses,
    uint32_t* out,
    unsigned long long count)
{
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    for(; i < count; i += stride)
    {
        const uint32_t table_id = table_ids[i];
        const uint32_t address = addresses[i];
        out[i] = __ldg(tables + table_id * entries + address);
    }
}

__global__ void pivot_v2_many_lut_kernel(
    const clut_pivot::DeviceV2Table* tables,
    const uint32_t* table_ids,
    const uint32_t* addresses,
    uint32_t* out,
    unsigned long long count)
{
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    for(; i < count; i += stride)
    {
        const clut_pivot::DeviceV2Table t = tables[table_ids[i]];
        out[i] = pivot_decode_v2_value(t, addresses[i]);
    }
}

__global__ void pivot_plain_llm_kernel(
    const uint32_t* tables,
    uint32_t entries,
    const uint32_t* group_ids,
    const uint32_t* group_to_table,
    const uint32_t* codes,
    const float* activations,
    float* out,
    unsigned long long count)
{
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    for(; i < count; i += stride)
    {
        const uint32_t table_id = group_to_table[group_ids[i]];
        const uint32_t value = __ldg(tables + table_id * entries + codes[i]);
        out[i] = activations[i] * static_cast<float>(value) * (1.0f / 4096.0f);
    }
}

__global__ void pivot_v2_llm_kernel(
    const clut_pivot::DeviceV2Table* tables,
    const uint32_t* group_ids,
    const uint32_t* group_to_table,
    const uint32_t* codes,
    const float* activations,
    float* out,
    unsigned long long count)
{
    unsigned long long i = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = gridDim.x * static_cast<unsigned long long>(blockDim.x);
    for(; i < count; i += stride)
    {
        const uint32_t table_id = group_to_table[group_ids[i]];
        const uint32_t value = pivot_decode_v2_value(tables[table_id], codes[i]);
        out[i] = activations[i] * static_cast<float>(value) * (1.0f / 4096.0f);
    }
}

#endif
