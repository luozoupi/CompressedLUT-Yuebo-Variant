#include "bench_common.h"

#include <cstdlib>
#include <thread>

namespace {

struct Options {
    std::string repo_root = ".";
    std::string out_path;
    size_t lookups = static_cast<size_t>(1) << 24;
    int repeats = 5;
    bool quick = false;
    std::vector<int> threads = {1, 2, 4, 8, 16, 32, 64, 128};
};

struct TimedResult {
    double median_ms = 0.0;
    uint64_t checksum = 0;
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
        else if(arg == "--threads" && i + 1 < argc)
            options.threads = clut_bench::parse_int_list(argv[++i]);
        else if(arg == "--quick")
        {
            options.quick = true;
            options.lookups = static_cast<size_t>(1) << 20;
            options.repeats = 2;
            options.threads = {1, 8};
        }
        else
            throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if(options.threads.empty())
        options.threads = {1};
    if(options.repeats < 1)
        options.repeats = 1;
    return options;
}

template <typename Lookup>
uint64_t run_parallel(size_t lookups, int thread_count, const Lookup& lookup)
{
    if(thread_count < 1)
        thread_count = 1;

    std::vector<std::thread> workers;
    std::vector<uint64_t> checksums(static_cast<size_t>(thread_count), 0);
    workers.reserve(static_cast<size_t>(thread_count));

    for(int t = 0; t < thread_count; t++)
    {
        const size_t begin = lookups * static_cast<size_t>(t) / static_cast<size_t>(thread_count);
        const size_t end = lookups * static_cast<size_t>(t + 1) / static_cast<size_t>(thread_count);
        workers.push_back(std::thread([&, t, begin, end]() {
            uint64_t local = 0;
            for(size_t i = begin; i < end; i++)
                local = clut_bench::mix_checksum(local, static_cast<uint64_t>(lookup(i)));
            checksums[static_cast<size_t>(t)] = local;
        }));
    }

    for(size_t i = 0; i < workers.size(); i++)
        workers[i].join();

    uint64_t checksum = 0;
    for(size_t i = 0; i < checksums.size(); i++)
        checksum += checksums[i];
    return checksum;
}

template <typename Lookup>
TimedResult measure(size_t lookups, int repeats, int threads, const Lookup& lookup)
{
    TimedResult result;
    result.checksum = run_parallel(lookups, threads, lookup);

    std::vector<double> times;
    times.reserve(static_cast<size_t>(repeats));
    for(int r = 0; r < repeats; r++)
    {
        const double start = clut_bench::now_ms();
        result.checksum = run_parallel(lookups, threads, lookup);
        const double end = clut_bench::now_ms();
        times.push_back(end - start);
    }
    result.median_ms = clut_bench::median(times);
    return result;
}

void emit_row(
    std::ostream& os,
    const clut_bench::Dataset& dataset,
    const clut_bench::CompressionResult& compression,
    const std::string& variant,
    const std::string& pattern,
    int threads,
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
        "cpu",
        variant,
        pattern,
        threads,
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

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
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
            std::cerr << "cpu: compressing " << dataset.name << " (" << dataset.table.size() << " entries)\n";
            const clut_bench::CompressionResult compression = clut_bench::compress_dataset(dataset);

            std::string validation_error;
            const bool decode_ok = clut_bench::validate_decode(dataset, compression.artifact, &validation_error);
            if(!decode_ok)
                std::cerr << "cpu: " << dataset.name << " decode validation failed: " << validation_error << "\n";

            for(size_t p = 0; p < 2; p++)
            {
                const std::string pattern = patterns[p];
                const std::vector<uint32_t> addresses = clut_bench::make_addresses(dataset.table.size(), options.lookups, pattern);

                for(size_t ti = 0; ti < options.threads.size(); ti++)
                {
                    const int threads = options.threads[ti];

                    const TimedResult plain = measure(options.lookups, options.repeats, threads, [&](size_t i) {
                        return dataset.table[addresses[i]];
                    });
                    emit_row(*out, dataset, compression, "plain_lut", pattern, threads, options.lookups, plain, "ok");

                    if(decode_ok)
                    {
                        const TimedResult clut = measure(options.lookups, options.repeats, threads, [&](size_t i) {
                            return compressedlut::decode(compression.artifact, addresses[i]);
                        });
                        emit_row(*out, dataset, compression, "compresslut", pattern, threads, options.lookups, clut, "ok");
                    }
                    else
                    {
                        TimedResult skipped;
                        emit_row(*out, dataset, compression, "compresslut", pattern, threads, options.lookups, skipped, validation_error);
                    }

                    if(dataset.function_kind != clut_bench::FunctionNone)
                    {
                        const double inv_entries = 1.0 / static_cast<double>(dataset.table.size());
                        const TimedResult libm = measure(options.lookups, options.repeats, threads, [&](size_t i) {
                            return clut_bench::quantize_function(dataset.function_kind, static_cast<double>(addresses[i]) * inv_entries, dataset.f_out);
                        });
                        emit_row(*out, dataset, compression, "libm", pattern, threads, options.lookups, libm, "ok");
                    }
                }
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "bench_cpu error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
