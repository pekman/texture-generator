#include <iostream>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <cmath>
#include "generate.hpp"

using namespace std;


int main(int argc, char *argv[])
{
    namespace po = boost::program_options;

    string cloudfile;
    string polygonfile;
    string outfile;
    unsigned texture_size;
    unsigned points_per_texel;
    float max_distance;
    float dist_power_param;

    po::options_description pos("Positional parameters");
    pos.add_options()
        ("cloudfile", po::value<string>(&cloudfile)->required())
        ("polygonfile", po::value<string>(&polygonfile)->required())
        ("outfile", po::value<string>(&outfile)->required())
        ;
    po::positional_options_description p;
    p.add("cloudfile", 1);
    p.add("polygonfile", 1);
    p.add("outfile", 1);

    po::options_description opts("Options");
    opts.add_options()
        ( "help,h", "show help" )
        ( "texture-size,s",
          po::value<unsigned>(&texture_size)
          ->default_value(256)->value_name("int"),
          "max testure size. The largest polygon by either width or height"
          " will have this as their texture width or height. Smaller"
          " polygons will have smaller textures." )
        ( "points-per-texel,k",
          po::value<unsigned>(&points_per_texel)
          ->default_value(1)->value_name("int"),
          "number of points per texel" )
        ( "max-distance,d",
          po::value<float>(&max_distance)
          ->default_value(INFINITY)->value_name("float"),
          "limit point search distance" )
        ( "dist-power-param,p",
          po::value<float>(&dist_power_param)
          ->default_value(3.0)->value_name("float"),
          "distance power parameter."
          " Greater values give greater influence to points closer to texel"
          " center. Should be >= 3.0." )
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
                 << " [options] cloudfile polygonfile outfile" << endl;
            cout << endl;
            cout << "  cloudfile      input point cloud file" << endl;
            cout << "  polygonfile    input polygon file" << endl;
            cout << "  outfile        output file name with or without extension"
                 << endl;
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

    if (! boost::algorithm::ends_with(outfile, ".x3d"))
        outfile += ".x3d";

    try {
        generate(
            cloudfile,
            polygonfile,
            outfile,
            texture_size,
            points_per_texel,
            max_distance * max_distance,
            dist_power_param);
    }
    catch (runtime_error &e) {
        cerr << e.what() << endl;
        return 126;
    }

    return 0;
}
