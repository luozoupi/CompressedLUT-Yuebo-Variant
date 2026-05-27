#include "../common/pivot_cuda_common.cuh"

#include <fstream>

namespace {

struct Options {
    std::string out_path;
    std::string group_counts;
    size_t lookups = static_cast<size_t>(1) << 24;
    int repeats = 5;
    int device = 7;
    int block_size = 256;
    int f_in = 12;
    int f_out = 12;
    int unique_tables = 16;
    bool quick = false;
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for(int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if(arg == "--out" && i + 1 < argc)
            options.out_path = argv[++i];
        else if(arg == "--group-counts" && i + 1 < argc)
            options.group_counts = argv[++i];
        else if(arg == "--unique-tables" && i + 1 < argc)
            options.unique_tables = std::stoi(argv[++i]);
        else if(arg == "--lookups" && i + 1 < argc)
            options.lookups = static_cast<size_t>(std::stoull(argv[++i]));
        else if(arg == "--repeats" && i + 1 < argc)
            options.repeats = std::stoi(argv[++i]);
        else if(arg == "--device" && i + 1 < argc)
            options.device = std::stoi(argv[++i]);
        else if(arg == "--block-size" && i + 1 < argc)
            options.block_size = std::stoi(argv[++i]);
        else if(arg == "--quick")
        {
            options.quick = true;
            options.lookups = static_cast<size_t>(1) << 22;
            options.repeats = 2;
        }
        else
            throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if(options.group_counts.empty())
        options.group_counts = options.quick ? "256,1024" : "256,1024,4096";
    return options;
}

uint32_t* upload_flat_tables(const std::vector<clut_bench::Dataset>& unique_datasets, size_t table_count)
{
    const size_t entries = unique_datasets.front().table.size();
    std::vector<uint32_t> flat(table_count * entries);
    for(size_t t = 0; t < table_count; t++)
    {
        const std::vector<long int>& source = unique_datasets[t % unique_datasets.size()].table;
        for(size_t i = 0; i < entries; i++)
            flat[t * entries + i] = static_cast<uint32_t>(source[i]);
    }

    uint32_t* ptr = NULL;
    PIVOT_CUDA_CHECK(cudaMalloc(&ptr, flat.size() * sizeof(uint32_t)));
    PIVOT_CUDA_CHECK(cudaMemcpy(ptr, flat.data(), flat.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return ptr;
}

std::vector<float> make_activations(size_t count)
{
    std::vector<float> values(count);
    std::mt19937 rng(24680);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(size_t i = 0; i < count; i++)
        values[i] = dist(rng);
    return values;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
        PIVOT_CUDA_CHECK(cudaSetDevice(options.device));
        cudaDeviceProp prop;
        PIVOT_CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
        std::cerr << "llm_lut: device " << options.device << " " << prop.name << " sm_" << prop.major << prop.minor << "\n";

        std::ofstream file;
        std::ostream* out = &std::cout;
        if(!options.out_path.empty())
        {
            file.open(options.out_path.c_str());
            if(!file)
                throw std::runtime_error("could not open output file: " + options.out_path);
            out = &file;
        }
        clut_pivot::write_pivot_header(*out);

        std::vector<clut_bench::Dataset> unique_datasets;
        std::vector<clut_bench::CompressionResult> unique_compressions;
        const size_t unique_count = static_cast<size_t>(options.unique_tables);
        for(size_t i = 0; i < unique_count; i++)
        {
            clut_bench::Dataset dataset = clut_pivot::make_variant_dataset("codebook_variant_" + std::to_string(i), options.f_in, options.f_out, static_cast<int>(i));
            clut_bench::CompressionResult compression = clut_bench::compress_dataset(dataset);
            std::string error;
            if(!clut_bench::validate_decode(dataset, compression.artifact, &error))
                throw std::runtime_error("unique codebook validation failed: " + error);
            unique_datasets.push_back(dataset);
            unique_compressions.push_back(compression);
        }

        std::vector<const compressedlut::CompressedTable*> unique_artifacts;
        long int unique_initial_bits = 0;
        double unique_compression_ms = 0.0;
        for(size_t i = 0; i < unique_count; i++)
        {
            unique_artifacts.push_back(&unique_compressions[i].artifact);
            unique_initial_bits += unique_compressions[i].artifact.initial_size;
            unique_compression_ms += unique_compressions[i].build_ms;
        }

        clut_pivot::HostV2TableSet v2_shared_tables;
        v2_shared_tables.upload(unique_artifacts);
        uint32_t* d_plain_shared_tables = upload_flat_tables(unique_datasets, unique_count);

        const std::vector<int> group_counts = clut_bench::parse_int_list(options.group_counts);
        const std::string patterns[] = {"sequential", "random"};
        const size_t entries = unique_datasets.front().table.size();
        const int grid = static_cast<int>((options.lookups + static_cast<size_t>(options.block_size) - 1) / static_cast<size_t>(options.block_size));

        const std::vector<float> activations = make_activations(options.lookups);
        float* d_activations = clut_pivot::copy_f32_to_device(activations);
        float* d_out = NULL;
        PIVOT_CUDA_CHECK(cudaMalloc(&d_out, options.lookups * sizeof(float)));

        for(size_t g = 0; g < group_counts.size(); g++)
        {
            const size_t group_count = static_cast<size_t>(group_counts[g]);
            const std::string case_name = std::to_string(group_count) + "_groups";
            std::cerr << "llm_lut: preparing " << case_name << "\n";

            uint32_t* d_plain_per_group_tables = upload_flat_tables(unique_datasets, group_count);
            const std::vector<uint32_t> per_group_map = clut_pivot::make_codebook_ids(group_count, group_count);
            const std::vector<uint32_t> shared_map = clut_pivot::make_codebook_ids(group_count, unique_count);
            uint32_t* d_per_group_map = clut_pivot::copy_u32_to_device(per_group_map);
            uint32_t* d_shared_map = clut_pivot::copy_u32_to_device(shared_map);

            const size_t per_group_plain_bytes = group_count * entries * sizeof(uint32_t);
            const size_t shared_plain_bytes = unique_count * entries * sizeof(uint32_t);
            const long int per_group_initial_bits = unique_compressions.front().artifact.initial_size * static_cast<long int>(group_count);

            for(size_t p = 0; p < 2; p++)
            {
                const std::string pattern = patterns[p];
                const std::vector<uint32_t> group_ids = clut_pivot::make_group_ids(group_count, options.lookups, pattern);
                const std::vector<uint32_t> codes = clut_bench::make_addresses(entries, options.lookups, pattern);
                uint32_t* d_group_ids = clut_pivot::copy_u32_to_device(group_ids);
                uint32_t* d_codes = clut_pivot::copy_u32_to_device(codes);

                const clut_pivot::TimedResult plain_per_group = clut_pivot::measure_cuda(
                    options.lookups,
                    options.repeats,
                    [&]() {
                        pivot_plain_llm_kernel<<<grid, options.block_size>>>(d_plain_per_group_tables, static_cast<uint32_t>(entries),
                            d_group_ids, d_per_group_map, d_codes, d_activations, d_out, options.lookups);
                        PIVOT_CUDA_CHECK(cudaGetLastError());
                    },
                    [&]() { return clut_pivot::checksum_f32_device(d_out, options.lookups); });
                clut_pivot::write_pivot_row(*out, "llm_lut", case_name, "plain_per_group", pattern, entries, group_count, unique_count,
                    options.lookups, plain_per_group, per_group_plain_bytes, v2_shared_tables.storage_bits,
                    v2_shared_tables.runtime_bytes, per_group_initial_bits, unique_compression_ms, "ok");

                const clut_pivot::TimedResult plain_shared = clut_pivot::measure_cuda(
                    options.lookups,
                    options.repeats,
                    [&]() {
                        pivot_plain_llm_kernel<<<grid, options.block_size>>>(d_plain_shared_tables, static_cast<uint32_t>(entries),
                            d_group_ids, d_shared_map, d_codes, d_activations, d_out, options.lookups);
                        PIVOT_CUDA_CHECK(cudaGetLastError());
                    },
                    [&]() { return clut_pivot::checksum_f32_device(d_out, options.lookups); });
                clut_pivot::write_pivot_row(*out, "llm_lut", case_name, "plain_shared", pattern, entries, group_count, unique_count,
                    options.lookups, plain_shared, shared_plain_bytes, v2_shared_tables.storage_bits,
                    v2_shared_tables.runtime_bytes, unique_initial_bits, unique_compression_ms, "ok");

                const clut_pivot::TimedResult v2_shared = clut_pivot::measure_cuda(
                    options.lookups,
                    options.repeats,
                    [&]() {
                        pivot_v2_llm_kernel<<<grid, options.block_size>>>(v2_shared_tables.d_descs, d_group_ids, d_shared_map,
                            d_codes, d_activations, d_out, options.lookups);
                        PIVOT_CUDA_CHECK(cudaGetLastError());
                    },
                    [&]() { return clut_pivot::checksum_f32_device(d_out, options.lookups); });
                clut_pivot::write_pivot_row(*out, "llm_lut", case_name, "compresslut_v2_shared", pattern, entries, group_count, unique_count,
                    options.lookups, v2_shared, shared_plain_bytes, v2_shared_tables.storage_bits,
                    v2_shared_tables.runtime_bytes, unique_initial_bits, unique_compression_ms, "ok");

                cudaFree(d_group_ids);
                cudaFree(d_codes);
            }

            cudaFree(d_per_group_map);
            cudaFree(d_shared_map);
            cudaFree(d_plain_per_group_tables);
        }

        cudaFree(d_activations);
        cudaFree(d_out);
        cudaFree(d_plain_shared_tables);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "bench_llm_lut error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
