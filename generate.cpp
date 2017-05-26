#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <pcl/io/auto_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <png.h>
#include "generate.hpp"


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


void generate(
    const std::string &cloudfile,
    const std::string &polygonfile,
    const std::string &outfile,
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
        throw std::runtime_error(std::string("Error opening ") + cloudfile);
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud);

    // load polygons
    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
    pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_points(new pcl::PointCloud<pcl::PointXYZ>);
    // if (pcl::io::load(polygonfile, *mesh) == -1) {
    if (pcl::io::loadPLYFile(polygonfile, *mesh) == -1)
        throw std::runtime_error(std::string("Error opening ") + polygonfile);
    pcl::fromPCLPointCloud2(mesh->cloud, *mesh_points);

    // open output file
    std::ofstream x3dfile(outfile, std::ios::binary);
    x3dfile << std::fixed;  // don't use scientific format for floats
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
    png_byte png_row[texture_size * 3];

    unsigned polygon_id = 1;

    size_t num_polygons = mesh->polygons.size();
    for (const pcl::Vertices &polygon : mesh->polygons) {
        std::cout << "processing polygon " << polygon_id
                  << "/" << num_polygons
                  << " (" << (100.0 * (polygon_id - 1) / num_polygons)
                  << "%)" << std::endl;


        // Calculate vectors for determining 3D location of each texel.
        // Use first two sides as v and u axes.

        Vector vertex1 = (*mesh_points)[polygon.vertices[0]];
        Vector vertex2 = (*mesh_points)[polygon.vertices[1]];
        Vector vertex3 = (*mesh_points)[polygon.vertices[2]];

        Vector v_side = vertex2 - vertex1;
        Vector u_side = vertex3 - vertex2;

        Vector u_unit = u_side.normalized();
        Vector v_unit = v_side.normalized();

        // If there are more than 3 vertices, check if any of them are
        // outside the parallelogram defined by the first 3 vertices,
        // and expand area as needed.
        double u_min = 0;
        double v_min = 0;
        double u_max = u_side.norm();
        double v_max = v_side.norm();
        for (auto it = polygon.vertices.cbegin() + 3;
             it != polygon.vertices.cend();  ++it)
        {
            Vector vertex = Vector((*mesh_points)[*it]) - vertex1;
            double scalarproj_u = u_unit.dot(vertex);
            double scalarproj_v = v_unit.dot(vertex);
            if (scalarproj_u > u_max) u_max = scalarproj_u;
            if (scalarproj_v > v_max) v_max = scalarproj_v;
            if (scalarproj_u < u_min) u_min = scalarproj_u;
            if (scalarproj_v < v_min) v_min = scalarproj_v;
        }

        Vector origin = vertex1 + u_unit*u_min + v_unit*v_min;
        Vector u_texelstep = (u_unit*(u_max - u_min)) / texture_size;
        Vector v_texelstep = (v_unit*(v_max - v_min)) / texture_size;


        // store polygon and its vertex and texture coordinates to X3D file

        Vector vertices[polygon.vertices.size()];
        for (size_t i=0; i < polygon.vertices.size(); ++i)
            vertices[i] = (*mesh_points)[polygon.vertices[i]];

        x3dfile <<
            "<Shape>"
            "<IndexedFaceSet coordIndex=\"0";
        for (size_t i=1; i < polygon.vertices.size(); ++i)
            x3dfile << ' ' << i;
        x3dfile << "\" texCoordIndex=\"0";
        for (size_t i=1; i < polygon.vertices.size(); ++i)
            x3dfile << ' ' << i;

        x3dfile << "\"><Coordinate point=\"" << vertices[0];
        for (size_t i=1; i < polygon.vertices.size(); ++i)
            x3dfile << ' ' << vertices[i];

        x3dfile << "\"/><TextureCoordinate point=\"";
        bool first = true;
        for (const Vector &vertex : vertices) {
            Vector relative_pos = vertex - origin;
            double u = u_unit.dot(relative_pos) / (u_max - u_min);
            double v = 1.0 - (v_unit.dot(relative_pos) / (v_max - v_min));

            if (! first)
                x3dfile << ' ';
            x3dfile << clamp01(u) << ' ' << clamp01(v);
            first = false;
        }

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
                                    std::string("Error opening ") + filename.str());
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

                if (num_found == 0) {
                    std::cerr << "warning: no points found near " << pos << std::endl;
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
                    for (unsigned i=0; i<num_found; ++i) {
                        float sqr_dist = point_sqr_distances[i];
                        if (sqr_dist <= max_sqr_dist) {
                            const pcl::PointXYZRGB &point = (*cloud)[point_indices[i]];

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
                    if (std::isfinite(r) && std::isfinite(g) && std::isfinite(b)) {
                        png_pixel[0] = std::lround(r);
                        png_pixel[1] = std::lround(g);
                        png_pixel[2] = std::lround(b);
                    }
                    else {
                        std::cerr << "warning: error in color calculation" << std::endl;
                        png_pixel[0] = 0;
                        png_pixel[1] = 0;
                        png_pixel[2] = 0;
                    }
                }
            }
            png_write_row(png_ptr, png_row);
        }

        png_write_end(png_ptr, NULL);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        if (std::fclose(fp) == EOF)
            throw std::system_error(errno, std::system_category(),
                                    std::string("Error closing ") + filename.str());

        polygon_id++;
    }

    x3dfile << "</Scene>\n</X3D>\n";
}
