Texture generator
=================

Generates textures for polygon scenes from a point cloud.

A texture is generated for each polygon in the polygon scene by using
k-nearest-neighbor search in the point cloud for each texel.


Build requirements
------------------

- [CMake] (≥3.0)
- [Boost] (≥1.58.0)
- [Point Cloud Library][PCL] (≥1.8.0)
- [Eigen] (≥3.3.2)
- [libpng] (≥1.2.54)


Compiling
---------

    cd texture-generator
    cmake .
    make


Running
-------

    texturegen [options] <point cloud file> <polygon scene file> <output file>

Point cloud file and polygon scene file can be in any format supported
by Point Cloud Library. Output file will be in X3D format. Textures
are generated as PNG files in the current directory. Output file
should have no path, or links to textures will be broken.

Run `texturegen --help` to see the options.


[CMake]: https://cmake.org/
[Boost]: http://www.boost.org/
[PCL]: http://pointclouds.org/
[Eigen]: http://eigen.tuxfamily.org/
[libpng]: http://www.libpng.org/
