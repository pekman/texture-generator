#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <algorithm>
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
#include <Eigen/LU>
#include <png.h>
#include "generate.hpp"

using std::size_t;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::isfinite;


const unsigned MIN_TEXTURE_SIZE = 16;


class internal_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};


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


// RAII-based PNG writer class. Separates gory libpng details from main code.
class PngWriter {
private:
    std::FILE *fp;
    png_structp png_ptr;
    png_infop info_ptr;
    const string filename;

public:
    PngWriter(
        unsigned polygon_id,
        unsigned texture_width, unsigned texture_height,
        bool alpha)
        : filename("texture" + std::to_string(polygon_id) + ".png")
    {
        // init png writing and write header

        fp = std::fopen(filename.c_str(), "wb");
        if (fp == NULL)
            throw std::system_error(
                errno, std::system_category(), "Error opening " + filename);

        png_ptr = png_create_write_struct(
            PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (png_ptr == NULL) {
            std::fclose(fp);
            throw std::runtime_error("Error: png_create_write_struct failed");
        }

        info_ptr = png_create_info_struct(png_ptr);
        if (info_ptr == NULL) {
            png_destroy_write_struct(&png_ptr, NULL);
            std::fclose(fp);
            throw std::runtime_error("Error: png_create_info_struct failed");
        }

        if (setjmp(png_jmpbuf(png_ptr))) {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            std::fclose(fp);
            throw std::runtime_error(
                "Error generating or writing png header to " + filename);
        }

        png_init_io(png_ptr, fp);
        png_set_IHDR(
            png_ptr, info_ptr,
            texture_width, texture_height,
            8,  // 8 bits per channel
            alpha ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_ptr, info_ptr);
    }

    ~PngWriter() {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        if (fp != NULL)
            std::fclose(fp);
    }

    void write_row(png_byte *data) {
        if (setjmp(png_jmpbuf(png_ptr)))
            throw std::runtime_error("Error writing png data to " + filename);

        png_write_row(png_ptr, data);
    }

    void write_end() {
        if (setjmp(png_jmpbuf(png_ptr)))
            throw std::runtime_error("Error writing png footer to " + filename);

        png_write_end(png_ptr, NULL);

        int status = std::fclose(fp);
        fp = NULL;
        if (status == EOF)
            throw std::system_error(
                errno, std::system_category(), "Error closing " + filename);
    }
};


// clamp texture coordinate between 0.0 and 1.0, because values
// outside that are rounding errors
static inline double clamp01(double val) {
    if (! (val > 0.0))  // eliminates <0, -0.0, and NaN
        return 0.0;
    if (val >= 1.0)
        return 1.0;
    return val;
}


// merge identical consecutive vertices
template <class T>
static inline void merge_identical_vertices(T &vertices) {
    if (vertices.size() < 3)
        throw internal_error("skipping polygon with fewer than 3 vertices");

    auto it = vertices.cbegin() + 1;
    do {
        if (it[0] == it[-1])
            it = vertices.erase(it);
        else
            ++it;
    } while (it != vertices.cend());
    if (! vertices.empty() && vertices.front() == vertices.back())
        vertices.pop_back();

    if (vertices.size() < 3)
        throw internal_error(
            "skipping polygon with too many identical vertices");
}


void generate(
    const string &cloudfile,
    const string &polygonfile,
    const string &outfile,
    unsigned max_texture_size,
    unsigned points_per_texel,
    float max_sqr_dist,
    float dist_power_param,
    bool backface)
{
    // smallest distance that can be safely used in inverse distance weighting
    double min_safe_dist =
        (256 * points_per_texel) / std::numeric_limits<double>::max();

    // if max_sqr_dist given, use alpha channel in PNG
    bool alpha = isfinite(max_sqr_dist);
    unsigned pixel_size = alpha ? 4 : 3;


    // load input point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::load(cloudfile, *cloud) == -1)
        throw std::runtime_error("Error opening " + cloudfile);
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud);

    // load polygons
    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
    pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_points(new pcl::PointCloud<pcl::PointXYZ>);
    // if (pcl::io::load(polygonfile, *mesh) == -1) {
    if (pcl::io::loadPLYFile(polygonfile, *mesh) == -1)
        throw std::runtime_error("Error opening " + polygonfile);
    pcl::fromPCLPointCloud2(mesh->cloud, *mesh_points);

    // find longest side of any polygon and store it as a base value
    // for texture size calculations (note: for polygons with more
    // than 4 sides, the value will probably be too small)
    double max_polygon_size = 0.0;
    for (const pcl::Vertices &polygon : mesh->polygons) {
        Vector previous = (*mesh_points)[polygon.vertices.back()];
        for (auto index : polygon.vertices) {
            Vector current = (*mesh_points)[index];
            double len = (current - previous).norm();
            if (len > max_polygon_size)
                max_polygon_size = len;
            previous = current;
        }
    }

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
        max_texture_size * pixel_size, boost::container::default_init);

    unsigned polygon_id = 1;

    size_t num_polygons = mesh->polygons.size();
    for (const pcl::Vertices &polygon : mesh->polygons) {
        cout << "processing polygon " << polygon_id << "/" << num_polygons
             << " (" << (100.0 * (polygon_id-1) / num_polygons) << "%)" << endl;

        try {
            // Calculate vectors for determining 3D location of each texel.
            // Use first two sides with length > 0 as v and u axes.

            boost::container::small_vector<Vector, 4> vertices;
            vertices.reserve(polygon.vertices.size());
            for (auto index : polygon.vertices)
                vertices.push_back((*mesh_points)[index]);

            merge_identical_vertices(vertices);
            Vector v_side = vertices[1] - vertices[0];
            Vector u_side = vertices[2] - vertices[1];

            Vector u_unit = u_side.normalized();
            Vector v_unit = v_side.normalized();

            // If there are more than 3 vertices, check if any of them
            // are outside the parallelogram defined by the first 3
            // vertices, and expand area as needed.
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
            double u_size = u_max - u_min;
            double v_size = v_max - v_min;

            // Calculate texture size from the size ratio of this
            // polygon and the largest one. Use powers of two as size
            // divisors favoring larger texture size.
            unsigned texture_width = std::max(
                MIN_TEXTURE_SIZE,
                max_texture_size >> int(
                    std::floor(std::log2(max_polygon_size / u_size))));
            unsigned texture_height = std::max(
                MIN_TEXTURE_SIZE,
                max_texture_size >> int(
                    std::floor(std::log2(max_polygon_size / v_size))));

            // Calculate vectors for calculating center of each
            // texel. For edge texels, the u or v distance from texel
            // center to the bounding parallelogram side is 0.5 *
            // texel width or height.
            Vector u_texelstep = (u_unit*u_size) / (texture_width + 1);
            Vector v_texelstep = (v_unit*v_size) / (texture_height + 1);
            Vector texel_origin =
                vertices[0] +
                u_unit*u_min + 0.5*u_texelstep +
                v_unit*v_min + 0.5*v_texelstep;


            // calculate texture coordinates

            boost::container::small_vector<float, 4*2> texturecoords;
            texturecoords.reserve(vertices.size() * 2);

            // Make transformation matrix for change of basis from
            // standard basis point cloud coordinates to u-v texture
            // coordinates. First, make inverse of the matrix. Set its
            // columns to vectors representing u-side and inverse
            // v-side of the parallelogram, and a dummy w vector.
            Eigen::Matrix3d basis_change_inv;
            basis_change_inv <<
                u_unit * u_size,
                -v_unit * v_size,
                // this can be anything orthogonal to both u_unit and v_unit:
                u_unit.cross(v_unit);
            Eigen::Matrix3d basis_change = basis_change_inv.inverse();

            Vector polygon_origin = vertices[0] + u_unit*u_min + v_unit*v_max;

            for (const Vector &vertex : vertices) {
                Vector relative_pos = vertex - polygon_origin;
                Vector texture_coord = basis_change * relative_pos;

                if (! isfinite(texture_coord.x()) || ! isfinite(texture_coord.y()))
                    throw internal_error(
                        "error generating texture coordinates, skipping polygon");

                texturecoords.push_back(clamp01(texture_coord.x()));
                texturecoords.push_back(clamp01(texture_coord.y()));
            }

            // store polygon and its vertex and texture coordinates to X3D file

            x3dfile <<
                "<Shape>"
                "<IndexedFaceSet coordIndex=\"0";
            for (size_t i=1; i < vertices.size(); ++i)
                x3dfile << ' ' << i;
            x3dfile << "\" texCoordIndex=\"0";
            for (size_t i=1; i < vertices.size(); ++i)
                x3dfile << ' ' << i;

            x3dfile << "\"><Coordinate point=\"" << vertices[0];
            for (size_t i=1; i < vertices.size(); ++i)
                x3dfile << ' ' << vertices[i];

            x3dfile << "\"/><TextureCoordinate point=\"" << texturecoords[0];
            for (size_t i=1; i < vertices.size() * 2; ++i)
                x3dfile << ' ' << texturecoords[i];

            x3dfile <<
                "\"/></IndexedFaceSet>"
                "<Appearance>"
                "<ImageTexture url=\"texture" << polygon_id << ".png\"/>"
                "</Appearance>"
                "</Shape>\n";

            if (backface) {
                // Add backface. Use same coordinates in opposite order.
                x3dfile <<
                    "<Shape>"
                    "<IndexedFaceSet coordIndex=\"";
                for (size_t i = vertices.size() - 1;  i > 0;  --i)
                    x3dfile << i << ' ';
                x3dfile << "0\" texCoordIndex=\"";
                for (size_t i = vertices.size() - 1;  i > 0;  --i)
                    x3dfile << i << ' ';

                x3dfile << "0\"><Coordinate point=\"" << vertices[0];
                for (size_t i=1; i < vertices.size(); ++i)
                    x3dfile << ' ' << vertices[i];

                x3dfile << "\"/><TextureCoordinate point=\"" << texturecoords[0];
                for (size_t i=1; i < vertices.size() * 2; ++i)
                    x3dfile << ' ' << texturecoords[i];

                x3dfile <<
                    "\"/></IndexedFaceSet>"
                    "<Appearance>"
                    "<ImageTexture url=\"texture" << polygon_id << ".png\"/>"
                    "</Appearance>"
                    "</Shape>\n";
            }


            // build texture using k-nearest-neighbor search

            // init png writing and write header
            PngWriter png(polygon_id, texture_width, texture_height, alpha);

            // generate texture
            for (unsigned y=0; y<texture_height; ++y) {
                Vector row_origin = texel_origin + v_texelstep*y;

                #pragma omp parallel for firstprivate(point_indices, point_sqr_distances)
                for (unsigned x=0; x<texture_width; ++x) {
                    png_byte *png_pixel = &png_row[x * pixel_size];
                    pcl::PointXYZRGB pos = Vector(row_origin + u_texelstep*x);

                    int num_found = kdtree.nearestKSearch(
                        pos, points_per_texel, point_indices, point_sqr_distances);

                    if (num_found <= 0) {
                        cerr << "warning: no points found for " << pos << endl;
                        png_pixel[0] = 0;
                        png_pixel[1] = 0;
                        png_pixel[2] = 0;
                        if (alpha)
                            png_pixel[3] = 0;
                    }
                    else {
                        // calculate texel color from point colors
                        // using inverse distance weighting
                        double r_sum = 0.0;
                        double g_sum = 0.0;
                        double b_sum = 0.0;
                        double weight_sum = 0.0;
                        for (int i=0; i<num_found; ++i) {
                            float sqr_dist = point_sqr_distances[i];
                            if (sqr_dist <= max_sqr_dist) {
                                const pcl::PointXYZRGB &point =
                                    (*cloud)[point_indices[i]];

                                double dist = std::max(
                                    min_safe_dist,
                                    std::pow<double>(
                                        sqr_dist, dist_power_param / 2.0));

                                r_sum += point.r / dist;
                                g_sum += point.g / dist;
                                b_sum += point.b / dist;
                                weight_sum += 1.0/dist;
                            }
                        }

                        if (alpha && weight_sum == 0.0) {
                            // None of points found were within distance limit.
                            // Add transparent texel.
                            png_pixel[0] = 0;
                            png_pixel[1] = 0;
                            png_pixel[2] = 0;
                            png_pixel[3] = 0;
                        }
                        else {
                            double r = r_sum / weight_sum;
                            double g = g_sum / weight_sum;
                            double b = b_sum / weight_sum;
                            if (isfinite(r) && isfinite(g) && isfinite(b)) {
                                png_pixel[0] = std::lround(r);
                                png_pixel[1] = std::lround(g);
                                png_pixel[2] = std::lround(b);
                                if (alpha)
                                    png_pixel[3] = 0xff;
                            }
                            else {
                                cerr << "warning: error in color calculation" << endl;
                                png_pixel[0] = 0;
                                png_pixel[1] = 0;
                                png_pixel[2] = 0;
                                if (alpha)
                                    png_pixel[3] = 0;
                            }
                        }
                    }
                }
                png.write_row(png_row.data());
            }

            png.write_end();
        }
        catch (internal_error &e) {
            cerr << "warning: " << e.what() << endl;
        }

        polygon_id++;
    }

    x3dfile << "</Scene>\n</X3D>\n";
}
