#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
using namespace glm;

struct AABB {
	float x_min{ FLT_MAX }, x_max{ -FLT_MAX };
	float y_min{ FLT_MAX }, y_max{ -FLT_MAX };
	float z_min{ FLT_MAX }, z_max{ -FLT_MAX };
	vec3 center{ 0.0f,0.0f,0.0f };
	void update(vec3 v) {
		x_min = std::min(v.x, x_min);
		x_max = std::max(v.x, x_max);
		y_min = std::min(v.y, y_min);
		y_max = std::max(v.y, y_max);
		z_min = std::min(v.z, z_min);
		z_max = std::max(v.z, z_max);
		center = { (x_max + x_min) / 2.0f,(y_max + y_min) / 2.0f,(z_max + z_min) / 2.0f };
	}
	void update(AABB v) {
		x_min = std::min(v.x_min, x_min);
		x_max = std::max(v.x_max, x_max);
		y_min = std::min(v.y_min, y_min);
		y_max = std::max(v.y_max, y_max);
		z_min = std::min(v.z_min, z_min);
		z_max = std::max(v.z_max, z_max);
		center = { (x_max + x_min) / 2.0f,(y_max + y_min) / 2.0f,(z_max + z_min) / 2.0f };
	}
};

struct Camera {
	std::string type;
	vec3 eye{ 0.0f,0.0f,0.0f };
	vec3 lookat{ 0.0f,0.0f,0.0f };
	vec3 up{ 0.0f,0.0f,0.0f };
	float fovy{ 0.0f };
	int width{ 0 }, height{ 0 };
};

struct Material {
	vec3 radiance{ 0.0f,0.0f,0.0f };
	vec3 Kd{ 0.0f,0.0f,0.0f };			// Diffuse color
	vec3 Ks{ 0.0f,0.0f,0.0f };			// Specular color
	float Ns{ 0.0f }, Ni{ 0.0f };		// Specular exponent & Optical density
	int map_id{-1};
};

struct Triangle {
	unsigned int v[3]{ 0,0,0 };
	unsigned int vt[3]{ 0,0,0 };
	unsigned int vn[3]{ 0,0,0 };
	std::string texture;
	AABB aabb;
};

struct MortonTri {
	unsigned int index{ 0 };
	AABB aabb;
	uint32_t MortonCode{ 0 };
};

//struct BVHNode {
//	void InitLeaf(int first, int n, const AABB& b) {
//		firstPrimOffset = first;
//		nPrimitives = n;
//		bounds = b;
//		children[0] = children[1] = nullptr;
//	}
//	void InitInterior(int axis, BVHNode* c0, BVHNode* c1) {
//		children[0] = c0;
//		children[1] = c1;
//		bounds.update(c0->bounds);
//		bounds.update(c1->bounds);
//		splitAxis = axis;
//		nPrimitives = 0;
//	}
//	AABB bounds;
//	BVHNode* children[2];
//	int splitAxis, firstPrimOffset, nPrimitives;
//};

//struct LBVHTreelet {
//	unsigned int startIndex, nPrimitives;
//	BVHNode* buildNodes;
//};

struct Triangle2Shader {
	vec3 v0, v1, v2;
	vec3 vn0, vn1, vn2;
	vec3 vt0, vt1, vt2;
	vec3 radiance;
	vec3 Kd;			// Diffuse color
	vec3 Ks;			// Specular color
	vec3 Ns_Ni_mapid;	// Specular exponent & Optical density & Texture id
};

struct BVHNode {
	int left, right;
	int n, index;              
	vec3 AA, BB;
};

struct BVHNode2Shader {
	vec3 childs;
	vec3 leafInfo;
	vec3 AA, BB;
};
