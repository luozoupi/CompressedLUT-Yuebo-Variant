#include "../common/pivot_cuda_common.cuh"

#include <fstream>

namespace {

struct Options {
    std::string out_path;
    std::string table_counts;
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
        else if(arg == "--table-counts" && i + 1 < argc)
            options.table_counts = argv[++i];
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
    if(options.table_counts.empty())
        options.table_counts = options.quick ? "16,64,256" : "16,64,256,1024";
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

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
        PIVOT_CUDA_CHECK(cudaSetDevice(options.device));
        cudaDeviceProp prop;
        PIVOT_CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
        std::cerr << "many_lut: device " << options.device << " " << prop.name << " sm_" << prop.major << prop.minor << "\n";

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
        unique_datasets.reserve(unique_count);
        unique_compressions.reserve(unique_count);
        for(size_t i = 0; i < unique_count; i++)
        {
            clut_bench::Dataset dataset = clut_pivot::make_variant_dataset("table_variant_" + std::to_string(i), options.f_in, options.f_out, static_cast<int>(i));
            clut_bench::CompressionResult compression = clut_bench::compress_dataset(dataset);
            std::string error;
            if(!clut_bench::validate_decode(dataset, compression.artifact, &error))
                throw std::runtime_error("unique table validation failed: " + error);
            unique_datasets.push_back(dataset);
            unique_compressions.push_back(compression);
        }

        const std::vector<int> table_counts = clut_bench::parse_int_list(options.table_counts);
        const std::string patterns[] = {"sequential", "random"};
        const size_t entries = unique_datasets.front().table.size();
        const int grid = static_cast<int>((options.lookups + static_cast<size_t>(options.block_size) - 1) / static_cast<size_t>(options.block_size));

        for(size_t c = 0; c < table_counts.size(); c++)
        {
            const size_t table_count = static_cast<size_t>(table_counts[c]);
            const size_t actual_unique = std::min(table_count, unique_count);
            const std::string case_name = std::to_string(table_count) + "x" + std::to_string(entries);
            std::cerr << "many_lut: preparing " << case_name << "\n";

            uint32_t* d_plain_tables = upload_flat_tables(unique_datasets, table_count);
            std::vector<const compressedlut::CompressedTable*> artifacts;
            artifacts.reserve(table_count);
            long int initial_bits = 0;
            double compression_ms = 0.0;
            for(size_t u = 0; u < actual_unique; u++)
                compression_ms += unique_compressions[u].build_ms;
            for(size_t t = 0; t < table_count; t++)
            {
                const clut_bench::CompressionResult& compression = unique_compressions[t % unique_count];
                artifacts.push_back(&compression.artifact);
                initial_bits += compression.artifact.initial_size;
            }

            clut_pivot::HostV2TableSet v2_tables;
            v2_tables.upload(artifacts);
            uint32_t* d_out = NULL;
            PIVOT_CUDA_CHECK(cudaMalloc(&d_out, options.lookups * sizeof(uint32_t)));
            const size_t plain_bytes = table_count * entries * sizeof(uint32_t);

            for(size_t p = 0; p < 2; p++)
            {
                const std::string pattern = patterns[p];
                const std::vector<uint32_t> group_ids = clut_pivot::make_group_ids(table_count, options.lookups, pattern);
                const std::vector<uint32_t> addresses = clut_bench::make_addresses(entries, options.lookups, pattern);
                uint32_t* d_group_ids = clut_pivot::copy_u32_to_device(group_ids);
                uint32_t* d_addresses = clut_pivot::copy_u32_to_device(addresses);

                const clut_pivot::TimedResult plain = clut_pivot::measure_cuda(
                    options.lookups,
                    options.repeats,
                    [&]() {
                        pivot_plain_many_lut_kernel<<<grid, options.block_size>>>(d_plain_tables, static_cast<uint32_t>(entries),
                            d_group_ids, d_addresses, d_out, options.lookups);
                        PIVOT_CUDA_CHECK(cudaGetLastError());
                    },
                    [&]() { return clut_pivot::checksum_u32_device(d_out, options.lookups); });
                clut_pivot::write_pivot_row(*out, "many_lut", case_name, "plain_lut", pattern, entries, table_count, actual_unique,
                    options.lookups, plain, plain_bytes, v2_tables.storage_bits, v2_tables.runtime_bytes,
                    initial_bits, compression_ms, "ok");

                const clut_pivot::TimedResult v2 = clut_pivot::measure_cuda(
                    options.lookups,
                    options.repeats,
                    [&]() {
                        pivot_v2_many_lut_kernel<<<grid, options.block_size>>>(v2_tables.d_descs, d_group_ids, d_addresses, d_out, options.lookups);
                        PIVOT_CUDA_CHECK(cudaGetLastError());
                    },
                    [&]() { return clut_pivot::checksum_u32_device(d_out, options.lookups); });
                clut_pivot::write_pivot_row(*out, "many_lut", case_name, "compresslut_v2", pattern, entries, table_count, actual_unique,
                    options.lookups, v2, plain_bytes, v2_tables.storage_bits, v2_tables.runtime_bytes,
                    initial_bits, compression_ms, "ok");

                cudaFree(d_group_ids);
                cudaFree(d_addresses);
            }

            cudaFree(d_out);
            cudaFree(d_plain_tables);
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "bench_many_lut error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
