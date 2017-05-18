#include <string>
#include <iostream>
// #include <fstream>
#include <cstdio>
#include <cmath>
#include <pcl/io/auto_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <png.h>


const int TEXTURE_SIZE = 1024;


png_structp png_ptr;
png_infop info_ptr;
png_byte png_row[TEXTURE_SIZE * 3];


class Vector : public Eigen::Vector3f {
public:
    Vector(const pcl::PointXYZ &p) : Vector(p.x, p.y, p.z) {}
    using Eigen::Vector3f::Vector3f;

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


int main(int argc, char *argv[])
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::load(argv[1], *cloud) == -1)
        return 2;

    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud);


    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
    pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_points(new pcl::PointCloud<pcl::PointXYZ>);
    // if (pcl::io::load(argv[1], *mesh) == -1) {
    if (pcl::io::loadPLYFile(argv[2], *mesh) == -1)
        return 2;
    pcl::fromPCLPointCloud2(mesh->cloud, *mesh_points);

    std::cout << cloud->size() << std::endl;
    std::cout << mesh->polygons.size() << std::endl;


    // std::ofstream objfile(argv[3], std::ios::binary);


    unsigned polygon_id = 1;

    size_t num_polygons = mesh->polygons.size();
    for (const pcl::Vertices &polygon : mesh->polygons) {
        std::cout << "processing polygon " << polygon_id
                  << "/" << num_polygons
                  << " (" << (100.0 * (polygon_id - 1) / num_polygons)
                  << "%)" << std::endl;

        Vector vertex1 = (*mesh_points)[polygon.vertices[0]];
        Vector vertex2 = (*mesh_points)[polygon.vertices[1]];
        Vector vertex3 = (*mesh_points)[polygon.vertices[2]];

        Vector v_side = vertex2 - vertex1;
        Vector u_side = vertex3 - vertex2;

        Vector u_unit = u_side.normalized();
        Vector v_unit = v_side.normalized();

        // If there are more than 3 vertices, check if any of them are
        // outside the parallelogram defined by the 3 first vertices,
        // and expand area as needed.
        float u_min = 0;
        float v_min = 0;
        float u_max = u_side.norm();
        float v_max = v_side.norm();
        for (auto it = polygon.vertices.cbegin() + 3;
             it != polygon.vertices.cend();  ++it)
        {
            Vector vertex = Vector((*mesh_points)[*it]) - vertex1;
            float scalarproj_u = u_unit.dot(vertex);
            float scalarproj_v = v_unit.dot(vertex);
            if (scalarproj_u > u_max) u_max = scalarproj_u;
            if (scalarproj_v > v_max) v_max = scalarproj_v;
            if (scalarproj_u < u_min) u_min = scalarproj_u;
            if (scalarproj_v < v_min) v_min = scalarproj_v;
        }

        Vector origin = vertex1 + u_unit*u_min + v_unit*v_min;
        Vector u_pixelstep = (u_unit*(u_max - u_min)) / TEXTURE_SIZE;
        Vector v_pixelstep = (v_unit*(v_max - v_min)) / TEXTURE_SIZE;


        // build texture using nearest-neighbor search

        // init png writing
        std::ostringstream filename;
        filename << "texture" << polygon_id++ << ".png";
        std::FILE *fp = std::fopen(filename.str().c_str(), "wb");
        if (! fp)
            return 126;
        // TODO: these should probably not be allocated and freed repeatedly
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png_ptr)
            return 126;
        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
            return 126;
        if (setjmp(png_jmpbuf(png_ptr)))
            return 126;
        png_init_io(png_ptr, fp);
        png_set_IHDR(
            png_ptr, info_ptr,
            TEXTURE_SIZE, TEXTURE_SIZE,  // width, height
            8, PNG_COLOR_TYPE_RGB,  // 8-bit RGB
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_ptr, info_ptr);

        std::vector<int> point_indices(1);
        std::vector<float> point_sqr_distances(1);

        // generate texture
        for (unsigned y=0; y<TEXTURE_SIZE; ++y) {
            Vector row_origin = origin + v_pixelstep*y;

            #pragma omp parallel for firstprivate(point_indices, point_sqr_distances)
            for (unsigned x=0; x<TEXTURE_SIZE; ++x) {
                png_byte *png_pixel = &png_row[x*3];
                pcl::PointXYZRGB pos = Vector(row_origin + u_pixelstep*x);
                int num_found =
                    kdtree.nearestKSearch(pos, 1, point_indices, point_sqr_distances);

                if (num_found == 0) {
                    std::cerr << "warning: no points found near(" << pos << std::endl;
                    png_pixel[0] = 0;
                    png_pixel[1] = 0;
                    png_pixel[2] = 0;
                }
                else {
                    const pcl::PointXYZRGB &point = (*cloud)[point_indices[0]];
                    png_pixel[0] = point.r;
                    png_pixel[1] = point.g;
                    png_pixel[2] = point.b;
                }
            }
            png_write_row(png_ptr, png_row);
        }

        png_write_end(png_ptr, NULL);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
    }

    return 0;
}
