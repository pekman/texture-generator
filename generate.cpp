#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <boost/container/vector.hpp>
#include <boost/container/small_vector.hpp>
#include <pcl/io/auto_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <png.h>
#include "generate.hpp"

using std::size_t;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::isfinite;


// vector class with type conversions from and to PCL point types
class Vector : public Eigen::Vector3d {
public:
    Vector() {}
    Vector(const pcl::PointXYZ &p) : Vector(p.x, p.y, p.z) {}
    using Eigen::Vector3d::Vector3d;

    operator pcl::PointXYZ() const {
        return pcl::PointXYZ(x(), y(), z());
    }

    operator pcl::PointXYZRGB() const {
        pcl::PointXYZRGB val;
        val.x = x();  val.y = y();  val.z = z();
        val.r = val.g = val.b = 0;
        return val;
    }
};

static inline std::ostream &operator<<(std::ostream &os, const Vector &v) {
    os << v.x() << ' ' << v.y() << ' ' << v.z();
    return os;
}


// clamp texture coordinate between 0.0 and 1.0, because values
// outside that are rounding errors
static inline double clamp01(double val) {
    if (! (val > 0.0))  // eliminates <0, -0.0, and NaN
        return 0.0;
    if (val >= 1.0)
        return 1.0;
    return val;
}


template <size_t N>
static inline bool any_two_equal(
    const boost::container::small_vector<Vector, N> &vertices)
{
    for (size_t i = 0; i < vertices.size() - 1; ++i)
        for (size_t j = i+1; j < vertices.size(); ++j)
            if (vertices[i] == vertices[j])
                return true;
    return false;
}


void generate(
    const string &cloudfile,
    const string &polygonfile,
    const string &outfile,
    unsigned texture_size,
    unsigned points_per_texel,
    float max_sqr_dist)
{
    // smallest distance that can be safely used in inverse distance weighting
    double min_safe_dist =
        (256 * points_per_texel) / std::numeric_limits<double>::max();


    // load input point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::load(cloudfile, *cloud) == -1)
        throw std::runtime_error(string("Error opening ") + cloudfile);
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud);

    // load polygons
    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
    pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_points(new pcl::PointCloud<pcl::PointXYZ>);
    // if (pcl::io::load(polygonfile, *mesh) == -1) {
    if (pcl::io::loadPLYFile(polygonfile, *mesh) == -1)
        throw std::runtime_error(string("Error opening ") + polygonfile);
    pcl::fromPCLPointCloud2(mesh->cloud, *mesh_points);

    // open output file
    std::ofstream x3dfile(outfile, std::ios::binary);
    // print floats with just enough digits for full precision
    x3dfile.precision(std::numeric_limits<float>::max_digits10);
    x3dfile <<
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE X3D PUBLIC \"ISO//Web3D//DTD X3D 3.3//EN\""
        " \"http://www.web3d.org/specifications/x3d-3.3.dtd\">\n"
        "<X3D profile=\"Interchange\" version=\"3.3\""
        " xmlns:xsd=\"http://www.w3.org/2001/XMLSchema-instance\""
        " xsd:noNamespaceSchemaLocation=\"http://www.web3d.org/specifications/x3d-3.2.xsd\">\n"
        "<Scene>";


    std::vector<int> point_indices(points_per_texel);
    std::vector<float> point_sqr_distances(points_per_texel);
    boost::container::vector<png_byte> png_row(
        texture_size * 3, boost::container::default_init);

    unsigned polygon_id = 1;

    size_t num_polygons = mesh->polygons.size();
    for (const pcl::Vertices &polygon : mesh->polygons) {
        cout << "processing polygon " << polygon_id << "/" << num_polygons
             << " (" << (100.0 * (polygon_id-1) / num_polygons) << "%)" << endl;


        // Calculate vectors for determining 3D location of each texel.
        // Use first two sides as v and u axes.

        size_t num_vertices = polygon.vertices.size();
        boost::container::small_vector<Vector, 4> vertices;
        vertices.reserve(num_vertices);
        for (auto index : polygon.vertices)
            vertices.push_back((*mesh_points)[index]);

        Vector v_side = vertices[1] - vertices[0];
        Vector u_side = vertices[2] - vertices[1];

        Vector u_unit = u_side.normalized();
        Vector v_unit = v_side.normalized();

        // If there are more than 3 vertices, check if any of them are
        // outside the parallelogram defined by the first 3 vertices,
        // and expand area as needed.
        double u_min = 0;
        double v_min = 0;
        double u_max = u_side.norm();
        double v_max = v_side.norm();
        for (auto it = vertices.cbegin() + 3; it != vertices.cend();  ++it) {
            Vector relative_pos = *it - vertices[0];
            double scalarproj_u = u_unit.dot(relative_pos);
            double scalarproj_v = v_unit.dot(relative_pos);
            if (scalarproj_u > u_max) u_max = scalarproj_u;
            if (scalarproj_v > v_max) v_max = scalarproj_v;
            if (scalarproj_u < u_min) u_min = scalarproj_u;
            if (scalarproj_v < v_min) v_min = scalarproj_v;
        }

        Vector origin = vertices[0] + u_unit*u_min + v_unit*v_min;
        Vector u_texelstep = (u_unit*(u_max - u_min)) / texture_size;
        Vector v_texelstep = (v_unit*(v_max - v_min)) / texture_size;


        // calculate texture coordinates

        bool shape_valid = true;
        boost::container::small_vector<float, 4*2> texturecoords;
        texturecoords.reserve(num_vertices * 2);

        for (const Vector &vertex : vertices) {
            Vector relative_pos = vertex - origin;
            double u = u_unit.dot(relative_pos) / (u_max - u_min);
            double v = 1.0 - (v_unit.dot(relative_pos) / (v_max - v_min));

            if (! isfinite(u) || ! isfinite(v)) {
                cerr <<
                    "warning: error generating texture coordinates,"
                    " skipping polygon" << endl;
                if (any_two_equal(vertices))
                    cerr << "         (probable cause: identical vertices)" << endl;

                shape_valid = false;
                break;
            }

            texturecoords.push_back(clamp01(u));
            texturecoords.push_back(clamp01(v));
        }

        if (shape_valid) {
            // store polygon and its vertex and texture coordinates to X3D file

            x3dfile <<
                "<Shape>"
                "<IndexedFaceSet coordIndex=\"0";
            for (size_t i=1; i < num_vertices; ++i)
                x3dfile << ' ' << i;
            x3dfile << "\" texCoordIndex=\"0";
            for (size_t i=1; i < num_vertices; ++i)
                x3dfile << ' ' << i;

            x3dfile << "\"><Coordinate point=\"" << vertices[0];
            for (size_t i=1; i < num_vertices; ++i)
                x3dfile << ' ' << vertices[i];

            x3dfile << "\"/><TextureCoordinate point=\"" << texturecoords[0];
            for (size_t i=1; i < num_vertices * 2; ++i)
                x3dfile << ' ' << texturecoords[i];

            x3dfile <<
                "\"/></IndexedFaceSet>"
                "<Appearance>"
                "<ImageTexture url=\"texture" << polygon_id << ".png\"/>"
                "</Appearance>"
                "</Shape>\n";


            // build texture using nearest-neighbor search

            // init png writing and write header
            std::ostringstream filename;
            filename << "texture" << polygon_id << ".png";
            std::FILE *fp = std::fopen(filename.str().c_str(), "wb");
            if (! fp)
                throw std::system_error(errno, std::system_category(),
                                        string("Error opening ") + filename.str());
            png_structp png_ptr = png_create_write_struct(
                PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
            if (! png_ptr)
                throw std::runtime_error("Error: png_create_write_struct failed");
            png_infop info_ptr = png_create_info_struct(png_ptr);
            if (! info_ptr)
                throw std::runtime_error("Error: png_create_info_struct failed");
            if (setjmp(png_jmpbuf(png_ptr)))
                throw std::runtime_error("Error: unknown png writing error");
            png_init_io(png_ptr, fp);
            png_set_IHDR(
                png_ptr, info_ptr,
                texture_size, texture_size,  // width, height
                8, PNG_COLOR_TYPE_RGB,  // 8-bit RGB
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
            png_write_info(png_ptr, info_ptr);

            // generate texture
            for (unsigned y=0; y<texture_size; ++y) {
                Vector row_origin = origin + v_texelstep*y;

                #pragma omp parallel for firstprivate(point_indices, point_sqr_distances)
                for (unsigned x=0; x<texture_size; ++x) {
                    png_byte *png_pixel = &png_row[x*3];
                    pcl::PointXYZRGB pos = Vector(row_origin + u_texelstep*x);

                    int num_found = kdtree.nearestKSearch(
                        pos, points_per_texel, point_indices, point_sqr_distances);

                    if (num_found <= 0) {
                        cerr << "warning: no points found near " << pos << endl;
                        png_pixel[0] = 0;
                        png_pixel[1] = 0;
                        png_pixel[2] = 0;
                    }
                    else {
                        // calculate weighted average of point colors
                        double r_sum = 0.0;
                        double g_sum = 0.0;
                        double b_sum = 0.0;
                        double weight_sum = 0.0;
                        for (int i=0; i<num_found; ++i) {
                            float sqr_dist = point_sqr_distances[i];
                            if (sqr_dist <= max_sqr_dist) {
                                const pcl::PointXYZRGB &point =
                                    (*cloud)[point_indices[i]];

                                // dist = distance^4
                                double dist = (double)sqr_dist * (double)sqr_dist;
                                if (dist < min_safe_dist)
                                    dist = min_safe_dist;

                                r_sum += point.r / dist;
                                g_sum += point.g / dist;
                                b_sum += point.b / dist;
                                weight_sum += 1.0/dist;
                            }
                        }

                        double r = r_sum / weight_sum;
                        double g = g_sum / weight_sum;
                        double b = b_sum / weight_sum;
                        if (isfinite(r) && isfinite(g) && isfinite(b)) {
                            png_pixel[0] = std::lround(r);
                            png_pixel[1] = std::lround(g);
                            png_pixel[2] = std::lround(b);
                        }
                        else {
                            cerr << "warning: error in color calculation" << endl;
                            png_pixel[0] = 0;
                            png_pixel[1] = 0;
                            png_pixel[2] = 0;
                        }
                    }
                }
                png_write_row(png_ptr, png_row.data());
            }

            png_write_end(png_ptr, NULL);
            png_destroy_write_struct(&png_ptr, &info_ptr);
            if (std::fclose(fp) == EOF)
                throw std::system_error(errno, std::system_category(),
                                        string("Error closing ") + filename.str());
        }

        polygon_id++;
    }

    x3dfile << "</Scene>\n</X3D>\n";
}
