#include "Model.h"
using namespace std;

Model::Model() {};

Model::~Model() {};

// Load model from .obj file
int Model::ReadOBJ(const string& file) {
	ifstream infile;
	infile.open(file, ios::in);
	if (!infile.is_open()) {
		cout << "Read .obj Error." << endl;
		return -1;
	}
	string data;
	string texture;
	while (getline(infile, data)) {
		istringstream is(data);
		string prefix;
		is >> prefix;

		if (prefix == "o") {		// Model name
			is >> name;
		}
		if (prefix == "v") {		// Vertex
			vec3 vertex;
			is >> vertex.x >> vertex.y >> vertex.z;
			aabb.update(vertex);
			Vertexs.push_back(vertex);
		}
		if (prefix == "vt") {		// Texture coordinate
			vec2 vt;
			is >> vt.x >> vt.y;
			Vts.push_back(vt);
		}
		if (prefix == "vn") {		// Vertex normal vector
			vec3 vn;
			is >> vn.x >> vn.y >> vn.z;
			Vns.push_back(vn);
		}
		if (prefix == "usemtl") {	// Texture name
			is >> texture;
		}
		if (prefix == "f") {		// Fragment
			Triangle Tri;
			Tri.texture = texture;
			for (int i = 0; i < 3; ++i) {
				char o;
				unsigned int v, vt, vn;
				is >> v >> o >> vt >> o >> vn;
				Tri.v[i] = v - 1;
				Tri.vt[i] = vt - 1;
				Tri.vn[i] = vn - 1;
			}
			Tris.push_back(Tri);
		}
	}

	for (int i = 0; i < (int)Tris.size(); ++i) {	// Update triangles' AABB bounding box
		AABB tri_aabb;
		for (int j = 0; j < 3; ++j) {
			unsigned int vid = Tris[i].v[j];
			tri_aabb.update(Vertexs[vid]);
		}
		Tris[i].aabb = tri_aabb;
		MortonTri morton_tri;
		morton_tri.index = i;
		morton_tri.aabb = tri_aabb;
		MortonTris.push_back(morton_tri);
	}
	for (int i = 0; i < (int)Tris.size(); ++i) {	// Update triangles' AABB bounding box
		center_aabb.update(MortonTris[i].aabb.center);
	}
	if (name == "") {
		name = file;
	}
	cout << "Successfully read file " << file << "." << endl;
	cout << "Model " << name << " has " << Vertexs.size() << " vertexs and " << Tris.size() << " fragments." << endl;
	string line(89, '-');
	cout << line << endl;
	return 0;
};

// Load material library from .mtl file
int Model::ReadMTL(const string& file) {
	ifstream infile;
	infile.open(file, ios::in);
	if (!infile.is_open()) {
		cout << "Read .mtl Error." << endl;
		return -1;
	}
	string data;
	string material_name;
	Material material;
	while (getline(infile, data)) {
		istringstream is(data);
		string prefix;
		is >> prefix;
		
		if (prefix == "newmtl") {		// Texture name
			is >> material_name;
			Materials[material_name] = material;
		}
		if (prefix == "Kd") {			// Diffuse color
			vec3 Kd;
			is >> Kd.x >> Kd.y >> Kd.z;
			Materials[material_name].Kd = Kd;
		}
		if (prefix == "Ks") {			// Specular color
			vec3 Ks;
			is >> Ks.x >> Ks.y >> Ks.z;
			Materials[material_name].Ks = Ks;
		}
		if (prefix == "Ns") {			// Specular exponent
			is >> Materials[material_name].Ns;
		}
		if (prefix == "Ni") {			// Optical density
			is >> Materials[material_name].Ni;
		}
		if (prefix == "map_Kd") {		// Kd map texture
			string texture;
			is >> texture;
			Texture.push_back(texture);
			Materials[material_name].map_id = (int)Texture.size() - 1;
		}
	}
	cout << "Successfully read file " << file << "." << endl;
	cout << "Material library " << file << " has " << Materials.size() << " materials." << endl;
	string line(89, '-');
	cout << line << endl;
	return 0;
}

// Load params from .xml file
int Model::ReadXML(const string& file) {
	TiXmlDocument lconfigXML;
	if (!lconfigXML.LoadFile(file.c_str())){
		cout << "Read .xml Error." << endl;
		return -1;
	}
	const TiXmlNode* Camera_node = NULL;
	const TiXmlNode* Light_node = NULL;
	const TiXmlNode* Camera_params = NULL;

	Camera_node = lconfigXML.RootElement();
	camera.type = Camera_node->ToElement()->Attribute("type");

	char o;
	Camera_params = Camera_node->IterateChildren("eye", Camera_params);
	string eye = Camera_params->ToElement()->Attribute("value");
	istringstream is_eye(eye);
	is_eye >> camera.eye.x >> o >> camera.eye.y >> o >> camera.eye.z;
	
	Camera_params = Camera_node->IterateChildren("lookat", Camera_params);
	string lookat = Camera_params->ToElement()->Attribute("value");
	istringstream is_lookat(lookat);
	is_lookat >> camera.lookat.x >> o >> camera.lookat.y >> o >> camera.lookat.z;

	Camera_params = Camera_node->IterateChildren("up", Camera_params);
	string up = Camera_params->ToElement()->Attribute("value");
	istringstream is_up(up);
	is_up >> camera.up.x >> o >> camera.up.y >> o >> camera.up.z;

	Camera_params = Camera_node->IterateChildren("fovy", Camera_params);
	string fovy = Camera_params->ToElement()->Attribute("value");
	istringstream is_fovy(fovy);
	is_fovy >> camera.fovy;

	Camera_params = Camera_node->IterateChildren("width", Camera_params);
	string width = Camera_params->ToElement()->Attribute("value");
	camera.width = atoi(width.c_str());

	Camera_params = Camera_node->IterateChildren("height", Camera_params);
	string height = Camera_params->ToElement()->Attribute("value");
	camera.height = atoi(height.c_str());

	cout << "Camera:\t" << "type:\t\t" << camera.type << endl;
	cout << "\t" << "eye:\t\t" << eye << endl;
	cout << "\t" << "lookat:\t\t" << lookat << endl;
	cout << "\t" << "up:\t\t" << up << endl;
	cout << "\t" << "fovy:\t\t" << fovy << endl;
	cout << "\t" << "width:\t\t" << width << endl;
	cout << "\t" << "height:\t\t" << height << endl;

	Light_node = lconfigXML.RootElement()->NextSiblingElement();

	while (Light_node != NULL) {
		string mtlname;
		vec3 light_radiance;
		mtlname = Light_node->ToElement()->Attribute("mtlname");
		string radiance = Light_node->ToElement()->Attribute("radiance");
		istringstream is_radiance(radiance);
		is_radiance >> light_radiance.r >> o >> light_radiance.g >> o >> light_radiance.b;
		Materials[mtlname].radiance = light_radiance;
		cout << "Light:\t" << "mtlname:\t" << mtlname << endl;
		cout << "\t" << "radiance:\t" << radiance << endl;
		Light_node = Light_node->NextSiblingElement();
	}
	
	string line(89, '-');
	cout << line << endl;
	return 0;
}

bool cmpx(const Triangle2Shader& t1, const Triangle2Shader& t2) {
	vec3 center1 = (t1.v0 + t1.v1 + t1.v2) / vec3(3, 3, 3);
	vec3 center2 = (t2.v0 + t2.v1 + t2.v2) / vec3(3, 3, 3);
	return center1.x < center2.x;
}
bool cmpy(const Triangle2Shader& t1, const Triangle2Shader& t2) {
	vec3 center1 = (t1.v0 + t1.v1 + t1.v2) / vec3(3, 3, 3);
	vec3 center2 = (t2.v0 + t2.v1 + t2.v2) / vec3(3, 3, 3);
	return center1.y < center2.y;
}
bool cmpz(const Triangle2Shader& t1, const Triangle2Shader& t2) {
	vec3 center1 = (t1.v0 + t1.v1 + t1.v2) / vec3(3, 3, 3);
	vec3 center2 = (t2.v0 + t2.v1 + t2.v2) / vec3(3, 3, 3);
	return center1.z < center2.z;
}

int Model::buildBVHwithSAH(int l, int r, int n, int depth) {
	if (l > r) return 0;

	Nodes.push_back(BVHNode());
	int id = (int)Nodes.size() - 1;
	Nodes[id].left = Nodes[id].right = Nodes[id].n = Nodes[id].index = 0;
	Nodes[id].AA = vec3(iINF, iINF, iINF);
	Nodes[id].BB = vec3(-iINF, -iINF, -iINF);


	for (int i = l; i <= r; i++) {

		float minx = std::min(Tris2Shader[i].v0.x, std::min(Tris2Shader[i].v1.x, Tris2Shader[i].v2.x));
		float miny = std::min(Tris2Shader[i].v0.y, std::min(Tris2Shader[i].v1.y, Tris2Shader[i].v2.y));
		float minz = std::min(Tris2Shader[i].v0.z, std::min(Tris2Shader[i].v1.z, Tris2Shader[i].v2.z));
		Nodes[id].AA.x = std::min(Nodes[id].AA.x, minx);
		Nodes[id].AA.y = std::min(Nodes[id].AA.y, miny);
		Nodes[id].AA.z = std::min(Nodes[id].AA.z, minz);

		float maxx = std::max(Tris2Shader[i].v0.x, std::max(Tris2Shader[i].v1.x, Tris2Shader[i].v2.x));
		float maxy = std::max(Tris2Shader[i].v0.y, std::max(Tris2Shader[i].v1.y, Tris2Shader[i].v2.y));
		float maxz = std::max(Tris2Shader[i].v0.z, std::max(Tris2Shader[i].v1.z, Tris2Shader[i].v2.z));
		Nodes[id].BB.x = std::max(Nodes[id].BB.x, maxx);
		Nodes[id].BB.y = std::max(Nodes[id].BB.y, maxy);
		Nodes[id].BB.z = std::max(Nodes[id].BB.z, maxz);
	}


	if ((r - l + 1) <= n || depth >= 64) {
		if (depth >= 64) {
			cout << "Depth of the SAH BVH is over 64! Please edit the stack of the fshader." << endl;
		}
		Nodes[id].n = r - l + 1;
		Nodes[id].index = l;
		return id;
	}

	float Cost = INF;
	int Axis = 0;
	int Split = (l + r) / 2;
	for (int axis = 0; axis < 3; axis++) {

		if (axis == 0) std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpx);
		if (axis == 1) std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpy);
		if (axis == 2) std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpz);


		std::vector<vec3> leftMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> leftMin(r - l + 1, vec3(INF, INF, INF));

		for (int i = l; i <= r; i++) {
			Triangle2Shader& t = Tris2Shader[i];
			int bias = (i == l) ? 0 : 1;  

			leftMax[i - l].x = std::max(leftMax[i - l - bias].x, std::max(t.v0.x, std::max(t.v1.x, t.v2.x)));
			leftMax[i - l].y = std::max(leftMax[i - l - bias].y, std::max(t.v0.y, std::max(t.v1.y, t.v2.y)));
			leftMax[i - l].z = std::max(leftMax[i - l - bias].z, std::max(t.v0.z, std::max(t.v1.z, t.v2.z)));

			leftMin[i - l].x = std::min(leftMin[i - l - bias].x, std::min(t.v0.x, std::min(t.v1.x, t.v2.x)));
			leftMin[i - l].y = std::min(leftMin[i - l - bias].y, std::min(t.v0.y, std::min(t.v1.y, t.v2.y)));
			leftMin[i - l].z = std::min(leftMin[i - l - bias].z, std::min(t.v0.z, std::min(t.v1.z, t.v2.z)));
		}


		std::vector<vec3> rightMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> rightMin(r - l + 1, vec3(INF, INF, INF));

		for (int i = r; i >= l; i--) {
			Triangle2Shader& t = Tris2Shader[i];
			int bias = (i == r) ? 0 : 1;  

			rightMax[i - l].x = std::max(rightMax[i - l + bias].x, std::max(t.v0.x, std::max(t.v1.x, t.v2.x)));
			rightMax[i - l].y = std::max(rightMax[i - l + bias].y, std::max(t.v0.y, std::max(t.v1.y, t.v2.y)));
			rightMax[i - l].z = std::max(rightMax[i - l + bias].z, std::max(t.v0.z, std::max(t.v1.z, t.v2.z)));

			rightMin[i - l].x = std::min(rightMin[i - l + bias].x, std::min(t.v0.x, std::min(t.v1.x, t.v2.x)));
			rightMin[i - l].y = std::min(rightMin[i - l + bias].y, std::min(t.v0.y, std::min(t.v1.y, t.v2.y)));
			rightMin[i - l].z = std::min(rightMin[i - l + bias].z, std::min(t.v0.z, std::min(t.v1.z, t.v2.z)));
		}


		float cost = INF;
		int split = l;
		for (int i = l; i <= r - 1; i++) {
			float lenx, leny, lenz;

			vec3 leftAA = leftMin[i - l];
			vec3 leftBB = leftMax[i - l];
			lenx = leftBB.x - leftAA.x;
			leny = leftBB.y - leftAA.y;
			lenz = leftBB.z - leftAA.z;
			float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
			float leftCost = leftS * (i - l + 1);

			vec3 rightAA = rightMin[i + 1 - l];
			vec3 rightBB = rightMax[i + 1 - l];
			lenx = rightBB.x - rightAA.x;
			leny = rightBB.y - rightAA.y;
			lenz = rightBB.z - rightAA.z;
			float rightS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
			float rightCost = rightS * (r - i);


			float totalCost = leftCost + rightCost;
			if (totalCost < cost) {
				cost = totalCost;
				split = i;
			}
		}

		if (cost < Cost) {
			Cost = cost;
			Axis = axis;
			Split = split;
		}
	}


	if (Axis == 0) std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpx);
	if (Axis == 1) std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpy);
	if (Axis == 2) std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpz);


	int left = buildBVHwithSAH(l, Split, n, depth + 1);
	int right = buildBVHwithSAH(Split + 1, r, n, depth + 1);

	Nodes[id].left = left;
	Nodes[id].right = right;

	return id;
}

int Model::buildBVH(int l, int r, int n) {
	if (l > r) return 0;

	Nodes.push_back(BVHNode());
	int id = (int)Nodes.size() - 1;
	Nodes[id].left = Nodes[id].right = Nodes[id].n = Nodes[id].index = 0;
	Nodes[id].AA = vec3(iINF, iINF, iINF);
	Nodes[id].BB = vec3(-iINF, -iINF, -iINF);


	for (int i = l; i <= r; i++) {

		float minx = std::min(Tris2Shader[i].v0.x, std::min(Tris2Shader[i].v1.x, Tris2Shader[i].v2.x));
		float miny = std::min(Tris2Shader[i].v0.y, std::min(Tris2Shader[i].v1.y, Tris2Shader[i].v2.y));
		float minz = std::min(Tris2Shader[i].v0.z, std::min(Tris2Shader[i].v1.z, Tris2Shader[i].v2.z));
		Nodes[id].AA.x = std::min(Nodes[id].AA.x, minx);
		Nodes[id].AA.y = std::min(Nodes[id].AA.y, miny);
		Nodes[id].AA.z = std::min(Nodes[id].AA.z, minz);

		float maxx = std::max(Tris2Shader[i].v0.x, std::max(Tris2Shader[i].v1.x, Tris2Shader[i].v2.x));
		float maxy = std::max(Tris2Shader[i].v0.y, std::max(Tris2Shader[i].v1.y, Tris2Shader[i].v2.y));
		float maxz = std::max(Tris2Shader[i].v0.z, std::max(Tris2Shader[i].v1.z, Tris2Shader[i].v2.z));
		Nodes[id].BB.x = std::max(Nodes[id].BB.x, maxx);
		Nodes[id].BB.y = std::max(Nodes[id].BB.y, maxy);
		Nodes[id].BB.z = std::max(Nodes[id].BB.z, maxz);
	}


	if ((r - l + 1) <= n) {
		Nodes[id].n = r - l + 1;
		Nodes[id].index = l;
		return id;
	}

	float lenx = Nodes[id].BB.x - Nodes[id].AA.x;
	float leny = Nodes[id].BB.y - Nodes[id].AA.y;
	float lenz = Nodes[id].BB.z - Nodes[id].AA.z;

	if (lenx >= leny && lenx >= lenz)
		std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpx);

	if (leny >= lenx && leny >= lenz)
		std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpy);

	if (lenz >= lenx && lenz >= leny)
		std::sort(Tris2Shader.begin() + l, Tris2Shader.begin() + r + 1, cmpz);

	int mid = (l + r) / 2;
	int left = buildBVH(l, mid, n);
	int right = buildBVH(mid + 1, r, n);

	Nodes[id].left = left;
	Nodes[id].right = right;

	return id;
}

void Model::Pre2Shader() {
	Tris2Shader.clear();
	for (int i = 0; i < (int)Tris.size(); ++i) {
		Triangle2Shader tri2shader;
		Material material = Materials[Tris[i].texture];
		tri2shader.v0 = Vertexs[Tris[i].v[0]];
		tri2shader.v1 = Vertexs[Tris[i].v[1]];
		tri2shader.v2 = Vertexs[Tris[i].v[2]];

		tri2shader.vn0 = Vns[Tris[i].vn[0]];
		tri2shader.vn1 = Vns[Tris[i].vn[1]];
		tri2shader.vn2 = Vns[Tris[i].vn[2]];

		tri2shader.vt0 = vec3(Vts[Tris[i].vt[0]], 0.0f);
		tri2shader.vt1 = vec3(Vts[Tris[i].vt[1]], 0.0f);
		tri2shader.vt2 = vec3(Vts[Tris[i].vt[2]], 0.0f);

		tri2shader.radiance = material.radiance;
		tri2shader.Kd = material.Kd;
		tri2shader.Ks = material.Ks;
		tri2shader.Ns_Ni_mapid = vec3(material.Ns, material.Ni, float(material.map_id));

		Tris2Shader.push_back(tri2shader);
		if (length(material.radiance) > 0.00001f) {
			Lights2Shader.push_back(tri2shader);
		}
	}
	buildBVHwithSAH(0, (int)Tris2Shader.size() - 1, 8);
	int nNodes = (int)Nodes.size();
	for (int i = 0; i < nNodes; i++) {
		BVHNode2Shader node;
		node.childs = vec3(Nodes[i].left, Nodes[i].right, 0);
		node.leafInfo = vec3(Nodes[i].n, Nodes[i].index, 0);
		node.AA = Nodes[i].AA;
		node.BB = Nodes[i].BB;
		Nodes2Shader.push_back(node);
	}
}

//__device__ uint32_t LeftShift3(uint32_t x) {
//	if (x == (1 << 10)) --x;
//	x = (x | (x << 16)) & 0x30000ff;
//	x = (x | (x << 8)) & 0x300f00f;
//	x = (x | (x << 4)) & 0x30c30c3;
//	x = (x | (x << 2)) & 0x9249249;
//	return x;
//}

//__device__ vec3 GetRelativePosition(const vec3& p, const AABB* aabb) {
//	vec3 pMin(aabb->x_min, aabb->y_min, aabb->z_min);
//	vec3 pMax(aabb->x_max, aabb->y_max, aabb->z_max);
//	vec3 o = p - pMin;
//	if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
//	if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
//	if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
//	return o;
//}
//
//__global__ void EncodeMorton3(MortonTri* Tris, const int* size, const AABB* aabb) {
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//	constexpr int mortonBits = 10;
//	constexpr int mortonScale = 1 << mortonBits;
//	vec3 RelativePosition = GetRelativePosition(Tris[index].aabb.center, aabb);
//	vec3 v = RelativePosition * (float)mortonScale;
//	Tris[index].MortonCode = (LeftShift3((uint32_t)v.z) << 2) | (LeftShift3((uint32_t)v.y) << 1) | LeftShift3((uint32_t)v.x);
//}

//void Model::GetMorton() {
//	MortonTri* tris;
//	int size = MortonTris.size();
//	
//	int* size_cuda;
//	AABB* center_aabb_cuda;
//	int nBytes = sizeof(MortonTri) * MortonTris.size();
//	double start = clock();
//	HANDLE_ERROR(cudaMalloc((void**)&tris, nBytes));
//	HANDLE_ERROR(cudaMalloc((void**)&size_cuda, sizeof(int)));
//	HANDLE_ERROR(cudaMalloc((void**)&center_aabb_cuda, sizeof(AABB)));
//
//	HANDLE_ERROR(cudaMemcpy(tris, &MortonTris[0], nBytes, cudaMemcpyHostToDevice));
//	HANDLE_ERROR(cudaMemcpy(size_cuda, &size, sizeof(int), cudaMemcpyHostToDevice));
//	HANDLE_ERROR(cudaMemcpy(center_aabb_cuda, &center_aabb, sizeof(AABB), cudaMemcpyHostToDevice));
//
//	dim3 blockSize(1024);
//	dim3 gridSize((Tris.size() + blockSize.x - 1) / blockSize.x);
//	EncodeMorton3 << < gridSize, blockSize >> > (tris, size_cuda, center_aabb_cuda);
//
//	double end = clock();
//	cout << end - start << "ms" << endl;
//	HANDLE_ERROR(cudaMemcpy(&MortonTris[0], tris, nBytes, cudaMemcpyDeviceToHost));
//	
//	cudaFree(tris);
//}

//vector<Triangle> Model::RadixSort() {
//	vector<MortonTri> tempVector(MortonTris.size());
//	constexpr int bitsPerPass = 6;
//	constexpr int nBits = 30;
//	constexpr int nPasses = nBits / bitsPerPass;
//	for (int pass = 0; pass < nPasses; ++pass) {
//		// Perform one pass of radix std::sort, std::sorting bitsPerPass bits
//		int lowBit = pass * bitsPerPass;
//
//		// Set in and out vector pointers for radix std::sort pass
//		vector<MortonTri>& in = (pass & 1) ? tempVector : MortonTris;
//		vector<MortonTri>& out = (pass & 1) ? MortonTris : tempVector;
//
//		// Count number of zero bits in array for current radix std::sort bit
//		constexpr int nBuckets = 1 << bitsPerPass;
//		int bucketCount[nBuckets] = { 0 };
//		constexpr int bitMask = (1 << bitsPerPass) - 1;
//		for (const MortonTri& mp : in) {
//			int bucket = (mp.MortonCode >> lowBit) & bitMask;
//			++bucketCount[bucket];
//		}
//
//		// Compute starting index in output array for each bucket
//		int outIndex[nBuckets];
//		outIndex[0] = 0;
//		for (int i = 1; i < nBuckets; ++i)
//			outIndex[i] = outIndex[i - 1] + bucketCount[i - 1];
//
//		// Store std::sorted values in output array
//		for (const MortonTri& mp : in) {
//			int bucket = (mp.MortonCode >> lowBit) & bitMask;
//			out[outIndex[bucket]++] = mp;
//		}
//	}
//	// Copy final result from tempVector, if needed
//	if (nPasses & 1)
//		swap(MortonTris, tempVector);
//
//	vector<Triangle> tempTris(Tris.size());
//	for (int i = 0; i < (int)MortonTris.size(); ++i) {
//		tempTris[i] = Tris[MortonTris[i].index];
//	}
//	return tempTris;
//}

//__device__ BVHNode* emitLBVH(BVHNode*& buildNodes,
//	MortonTri* morton_tri, int nPrimitives, int* totalNodes,
//	vector<Triangle>& orderedPrims, atomic<int>* orderedPrimsOffset, int bitIndex) {
//
//}

//ParallelFor(
//	[&](int i) {
//	<< Generate ith LBVH treelet >>
//	int nodesCreated = 0;
//	const int firstBitIndex = 29 - 12;
//	LBVHTreelet& tr = treeletsToBuild[i];
//	tr.buildNodes =
//		emitLBVH(tr.buildNodes, primitiveInfo, &mortonPrims[tr.startIndex],
//			tr.nPrimitives, &nodesCreated, orderedPrims,
//			&orderedPrimsOffset, firstBitIndex);
//	atomicTotal += nodesCreated;
//
//}, treeletsToBuild.size());
//__global__ void BulidTreelet_cuda() {
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//}

//BVHNode* Model::buildUpperSAH(vector<BVHNode*>& treeletRoots, int start, int end, int* totalNodes) {
//	return treeletRoots[0];
//}

//BVHNode* Model::HLBVHBuild(int* totalNodes) {
//	GetMorton();
//	vector<Triangle> orderedTris = RadixSort();
//	vector<LBVHTreelet> treeletsToBuild;
//	for (int start = 0, end = 1; end <= (int)MortonTris.size(); ++end) {
//		uint32_t mask = 0b00111111111111000000000000000000;
//		if (end == (int)MortonTris.size() ||
//			((MortonTris[start].MortonCode & mask) !=
//				(MortonTris[end].MortonCode & mask))) {
//			// Add entry to treeletsToBuild for this treelet
//			int nPrimitives = end - start;
//			int maxBVHNodes = 2 * nPrimitives - 1;
//			//BVHNode* nodes = arena.Alloc<BVHNode>(maxBVHNodes, false);
//			BVHNode* nodes = new BVHNode[maxBVHNodes];
//			LBVHTreelet Treelet{ start, nPrimitives, nodes };
//			treeletsToBuild.push_back(Treelet);
//			start = end;
//		}
//	}
//
//	// Create LBVHs for treelets in parallel
//	atomic<int> atomicTotal(0), orderedPrimsOffset(0);
//	orderedTris.resize(Tris.size());
//	// Generate ith LBVH treelet
//	*totalNodes = atomicTotal;
//
//	// Create and return SAH BVH from LBVH treelets
//	vector<BVHNode*> finishedTreelets;
//	for (LBVHTreelet& treelet : treeletsToBuild)
//		finishedTreelets.push_back(treelet.buildNodes);
//	return buildUpperSAH(finishedTreelets, 0, finishedTreelets.size(), totalNodes);
//}
