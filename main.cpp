#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include "generate.hpp"

using namespace std;


int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    string cloudfile;
    string polygonfile;
    string outfile_base;
    unsigned texture_size;
    unsigned points_per_texel;
    float max_distance;

    po::options_description pos("Positional parameters");
    pos.add_options()
        ("cloudfile", po::value<string>(&cloudfile)->required())
        ("polygonfile", po::value<string>(&polygonfile)->required())
        ("outfile-base", po::value<string>(&outfile_base)->required())
        ;
    po::positional_options_description p;
    p.add("cloudfile", 1);
    p.add("polygonfile", 1);
    p.add("outfile-base", 1);

    po::options_description opts("Options");
    opts.add_options()
        ( "help,h", "show help" )
        ( "texture-size,s",
          po::value<unsigned>(&texture_size)->default_value(256),
          "texture size (width and height)")
        ( "points-per-texel,k",
          po::value<unsigned>(&points_per_texel)->default_value(1),
          "number of points per texel" )
        ( "max-distance,d",
          po::value<float>(&max_distance)->default_value(INFINITY),
          "limit point search distance" )
        ;

    try {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                  .options(po::options_description().add(opts).add(pos))
                  .positional(p)
                  .run(),
                  vm);

        if (argc == 1 || vm.count("help")) {
            cout << "Usage:  " << argv[0]
                 << " [options] cloudfile polygonfile outfile-base" << endl;
            cout << endl;
            cout << "  cloudfile       input point cloud file" << endl;
            cout << "  polygonfile     input polygon file" << endl;
            cout << "  outfile-base    base name (without extension) for output files" << endl;
            cout << endl;
            cout << opts << endl;
            return vm.count("help") ? 0 : 2;
        }

        po::notify(vm);
    }
    catch (po::error &e) {
        cerr << e.what() << endl;
        return 2;
    }

    try {
        generate(
            cloudfile,
            polygonfile,
            outfile_base,
            texture_size,
            points_per_texel,
            max_distance * max_distance);
    }
    catch (runtime_error &e) {
        cerr << e.what() << endl;
        return 126;
    }

    return 0;
}