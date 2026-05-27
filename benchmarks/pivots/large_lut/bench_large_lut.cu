#include "../common/pivot_cuda_common.cuh"

#include <fstream>

namespace {

struct FunctionSpec {
    std::string name;
    clut_bench::FunctionKind kind;
};

struct Options {
    std::string out_path;
    std::string f_ins;
    std::string functions = "sqrt,sigmoid";
    size_t lookups = static_cast<size_t>(1) << 24;
    int repeats = 5;
    int device = 7;
    int block_size = 256;
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
        else if(arg == "--f-ins" && i + 1 < argc)
            options.f_ins = argv[++i];
        else if(arg == "--functions" && i + 1 < argc)
            options.functions = argv[++i];
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
    if(options.f_ins.empty())
        options.f_ins = options.quick ? "12,16" : "12,16,18";
    return options;
}

std::vector<FunctionSpec> parse_functions(const std::string& value)
{
    std::vector<FunctionSpec> result;
    std::stringstream ss(value);
    std::string item;
    while(std::getline(ss, item, ','))
    {
        if(item == "sqrt")
            result.push_back({"sqrt", clut_bench::FunctionSqrt});
        else if(item == "sigmoid")
            result.push_back({"sigmoid", clut_bench::FunctionSigmoid});
        else if(item == "exp")
            result.push_back({"exp", clut_bench::FunctionExpShift});
        else if(item == "tanh")
            result.push_back({"tanh", clut_bench::FunctionTanh});
        else if(!item.empty())
            throw std::runtime_error("unknown function: " + item);
    }
    return result;
}

uint32_t* upload_plain_table(const std::vector<long int>& values)
{
    std::vector<uint32_t> packed(values.size());
    for(size_t i = 0; i < values.size(); i++)
        packed[i] = static_cast<uint32_t>(values[i]);

    uint32_t* ptr = NULL;
    PIVOT_CUDA_CHECK(cudaMalloc(&ptr, packed.size() * sizeof(uint32_t)));
    PIVOT_CUDA_CHECK(cudaMemcpy(ptr, packed.data(), packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
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
        std::cerr << "large_lut: device " << options.device << " " << prop.name << " sm_" << prop.major << prop.minor << "\n";

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

        const std::vector<int> f_ins = clut_bench::parse_int_list(options.f_ins);
        const std::vector<FunctionSpec> functions = parse_functions(options.functions);
        const std::string patterns[] = {"sequential", "random"};

        for(size_t f = 0; f < functions.size(); f++)
        {
            for(size_t s = 0; s < f_ins.size(); s++)
            {
                const int f_in = f_ins[s];
                const std::string case_name = functions[f].name + "_f" + std::to_string(f_in);
                std::cerr << "large_lut: compressing " << case_name << "\n";
                const clut_bench::Dataset dataset = clut_pivot::make_function_dataset(case_name, functions[f].kind, f_in, 12);
                const clut_bench::CompressionResult compression = clut_bench::compress_dataset(dataset);
                std::string error;
                const bool decode_ok = clut_bench::validate_decode(dataset, compression.artifact, &error);
                if(!decode_ok)
                    std::cerr << "large_lut: validation failed: " << error << "\n";

                clut_pivot::HostV2Table v2_table;
                if(decode_ok)
                    v2_table.upload(compression.artifact);

                uint32_t* d_plain_table = upload_plain_table(dataset.table);
                uint32_t* d_out = NULL;
                PIVOT_CUDA_CHECK(cudaMalloc(&d_out, options.lookups * sizeof(uint32_t)));
                const int grid = static_cast<int>((options.lookups + static_cast<size_t>(options.block_size) - 1) / static_cast<size_t>(options.block_size));
                const size_t plain_bytes = dataset.table.size() * sizeof(uint32_t);
                const long int initial_bits = compression.artifact.initial_size;

                for(size_t p = 0; p < 2; p++)
                {
                    const std::string pattern = patterns[p];
                    const std::vector<uint32_t> addresses = clut_bench::make_addresses(dataset.table.size(), options.lookups, pattern);
                    uint32_t* d_addresses = clut_pivot::copy_u32_to_device(addresses);

                    const clut_pivot::TimedResult plain = clut_pivot::measure_cuda(
                        options.lookups,
                        options.repeats,
                        [&]() {
                            pivot_plain_lut_kernel<<<grid, options.block_size>>>(d_plain_table, d_addresses, d_out, options.lookups);
                            PIVOT_CUDA_CHECK(cudaGetLastError());
                        },
                        [&]() { return clut_pivot::checksum_u32_device(d_out, options.lookups); });
                    clut_pivot::write_pivot_row(*out, "large_lut", case_name, "plain_lut", pattern, dataset.table.size(), 1, 1,
                        options.lookups, plain, plain_bytes, clut_bench::final_compressed_bits(compression.artifact),
                        v2_table.runtime_bytes, initial_bits, compression.build_ms, "ok");

                    if(decode_ok)
                    {
                        const clut_pivot::TimedResult v2 = clut_pivot::measure_cuda(
                            options.lookups,
                            options.repeats,
                            [&]() {
                                pivot_v2_lut_kernel<<<grid, options.block_size>>>(v2_table.d_desc, d_addresses, d_out, options.lookups);
                                PIVOT_CUDA_CHECK(cudaGetLastError());
                            },
                            [&]() { return clut_pivot::checksum_u32_device(d_out, options.lookups); });
                        clut_pivot::write_pivot_row(*out, "large_lut", case_name, "compresslut_v2", pattern, dataset.table.size(), 1, 1,
                            options.lookups, v2, plain_bytes, clut_bench::final_compressed_bits(compression.artifact),
                            v2_table.runtime_bytes, initial_bits, compression.build_ms, "ok");
                    }
                    else
                    {
                        clut_pivot::TimedResult skipped;
                        clut_pivot::write_pivot_row(*out, "large_lut", case_name, "compresslut_v2", pattern, dataset.table.size(), 1, 1,
                            options.lookups, skipped, plain_bytes, clut_bench::final_compressed_bits(compression.artifact),
                            v2_table.runtime_bytes, initial_bits, compression.build_ms, error);
                    }

                    cudaFree(d_addresses);
                }

                cudaFree(d_out);
                cudaFree(d_plain_table);
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "bench_large_lut error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
