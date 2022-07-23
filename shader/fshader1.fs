#version 440 core
layout(location = 0) out vec4 update;

in vec3 pos;

uniform int frameCounter;
uniform int ntextures;
uniform int ntriangles;
uniform int nlights;

uniform samplerBuffer Tris;
uniform samplerBuffer Tree;
uniform samplerBuffer Lights;

uniform sampler2D lastframe;

// Camera Params
uniform vec3 eye;
uniform mat4 view;
uniform float fovy;
uniform int width;
uniform int height;

// Texture
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D texture3;
uniform sampler2D texture4;
uniform sampler2D texture5;
uniform sampler2D texture6;
uniform sampler2D texture7;

#define INF                 55555555.0f
#define PI                  3.1415926535897f
#define INV_PI              0.31830988618f
#define SIZE_TRIANGLE       13
#define SIZE_BVHNODE        4
#define DL_SAMPLES          1
#define DEPTH				5

#define RED                 vec3(1.0,0.0,0.0)
#define YELLOW              vec3(1.0,1.0,0.0)
#define BLUE                vec3(0.0,0.0,1.0)
#define GREEN               vec3(0.0,1.0,0.0)

uint vvv[256];

void init_vvv() {
    vvv[0] = 2147483648u; vvv[1] = 1073741824u; vvv[2] = 536870912u; vvv[3] = 268435456u; vvv[4] = 134217728u; vvv[5] = 67108864u; vvv[6] = 33554432u; vvv[7] = 16777216u; vvv[8] = 8388608u; vvv[9] = 4194304u; vvv[10] = 2097152u; vvv[11] = 1048576u; vvv[12] = 524288u; vvv[13] = 262144u; vvv[14] = 131072u; vvv[15] = 65536u; vvv[16] = 32768u; vvv[17] = 16384u; vvv[18] = 8192u; vvv[19] = 4096u; vvv[20] = 2048u; vvv[21] = 1024u; vvv[22] = 512u; vvv[23] = 256u; vvv[24] = 128u; vvv[25] = 64u; vvv[26] = 32u; vvv[27] = 16u; vvv[28] = 8u; vvv[29] = 4u; vvv[30] = 2u; vvv[31] = 1u;
    vvv[32] = 2147483648u; vvv[33] = 3221225472u; vvv[34] = 2684354560u; vvv[35] = 4026531840u; vvv[36] = 2281701376u; vvv[37] = 3422552064u; vvv[38] = 2852126720u; vvv[39] = 4278190080u; vvv[40] = 2155872256u; vvv[41] = 3233808384u; vvv[42] = 2694840320u; vvv[43] = 4042260480u; vvv[44] = 2290614272u; vvv[45] = 3435921408u; vvv[46] = 2863267840u; vvv[47] = 4294901760u; vvv[48] = 2147516416u; vvv[49] = 3221274624u; vvv[50] = 2684395520u; vvv[51] = 4026593280u; vvv[52] = 2281736192u; vvv[53] = 3422604288u; vvv[54] = 2852170240u; vvv[55] = 4278255360u; vvv[56] = 2155905152u; vvv[57] = 3233857728u; vvv[58] = 2694881440u; vvv[59] = 4042322160u; vvv[60] = 2290649224u; vvv[61] = 3435973836u; vvv[62] = 2863311530u; vvv[63] = 4294967295u;
    vvv[64] = 2147483648u; vvv[65] = 3221225472u; vvv[66] = 1610612736u; vvv[67] = 2415919104u; vvv[68] = 3892314112u; vvv[69] = 1543503872u; vvv[70] = 2382364672u; vvv[71] = 3305111552u; vvv[72] = 1753219072u; vvv[73] = 2629828608u; vvv[74] = 3999268864u; vvv[75] = 1435500544u; vvv[76] = 2154299392u; vvv[77] = 3231449088u; vvv[78] = 1626210304u; vvv[79] = 2421489664u; vvv[80] = 3900735488u; vvv[81] = 1556135936u; vvv[82] = 2388680704u; vvv[83] = 3314585600u; vvv[84] = 1751705600u; vvv[85] = 2627492864u; vvv[86] = 4008611328u; vvv[87] = 1431684352u; vvv[88] = 2147543168u; vvv[89] = 3221249216u; vvv[90] = 1610649184u; vvv[91] = 2415969680u; vvv[92] = 3892340840u; vvv[93] = 1543543964u; vvv[94] = 2382425838u; vvv[95] = 3305133397u;
    vvv[96] = 2147483648u; vvv[97] = 3221225472u; vvv[98] = 536870912u; vvv[99] = 1342177280u; vvv[100] = 4160749568u; vvv[101] = 1946157056u; vvv[102] = 2717908992u; vvv[103] = 2466250752u; vvv[104] = 3632267264u; vvv[105] = 624951296u; vvv[106] = 1507852288u; vvv[107] = 3872391168u; vvv[108] = 2013790208u; vvv[109] = 3020685312u; vvv[110] = 2181169152u; vvv[111] = 3271884800u; vvv[112] = 546275328u; vvv[113] = 1363623936u; vvv[114] = 4226424832u; vvv[115] = 1977167872u; vvv[116] = 2693105664u; vvv[117] = 2437829632u; vvv[118] = 3689389568u; vvv[119] = 635137280u; vvv[120] = 1484783744u; vvv[121] = 3846176960u; vvv[122] = 2044723232u; vvv[123] = 3067084880u; vvv[124] = 2148008184u; vvv[125] = 3222012020u; vvv[126] = 537002146u; vvv[127] = 1342505107u;
    vvv[128] = 2147483648u; vvv[129] = 1073741824u; vvv[130] = 536870912u; vvv[131] = 2952790016u; vvv[132] = 4160749568u; vvv[133] = 3690987520u; vvv[134] = 2046820352u; vvv[135] = 2634022912u; vvv[136] = 1518338048u; vvv[137] = 801112064u; vvv[138] = 2707423232u; vvv[139] = 4038066176u; vvv[140] = 3666345984u; vvv[141] = 1875116032u; vvv[142] = 2170683392u; vvv[143] = 1085997056u; vvv[144] = 579305472u; vvv[145] = 3016343552u; vvv[146] = 4217741312u; vvv[147] = 3719483392u; vvv[148] = 2013407232u; vvv[149] = 2617981952u; vvv[150] = 1510979072u; vvv[151] = 755882752u; vvv[152] = 2726789248u; vvv[153] = 4090085440u; vvv[154] = 3680870432u; vvv[155] = 1840435376u; vvv[156] = 2147625208u; vvv[157] = 1074478300u; vvv[158] = 537900666u; vvv[159] = 2953698205u;
    vvv[160] = 2147483648u; vvv[161] = 1073741824u; vvv[162] = 1610612736u; vvv[163] = 805306368u; vvv[164] = 2818572288u; vvv[165] = 335544320u; vvv[166] = 2113929216u; vvv[167] = 3472883712u; vvv[168] = 2290089984u; vvv[169] = 3829399552u; vvv[170] = 3059744768u; vvv[171] = 1127219200u; vvv[172] = 3089629184u; vvv[173] = 4199809024u; vvv[174] = 3567124480u; vvv[175] = 1891565568u; vvv[176] = 394297344u; vvv[177] = 3988799488u; vvv[178] = 920674304u; vvv[179] = 4193267712u; vvv[180] = 2950604800u; vvv[181] = 3977188352u; vvv[182] = 3250028032u; vvv[183] = 129093376u; vvv[184] = 2231568512u; vvv[185] = 2963678272u; vvv[186] = 4281226848u; vvv[187] = 432124720u; vvv[188] = 803643432u; vvv[189] = 1633613396u; vvv[190] = 2672665246u; vvv[191] = 3170194367u;
    vvv[192] = 2147483648u; vvv[193] = 3221225472u; vvv[194] = 2684354560u; vvv[195] = 3489660928u; vvv[196] = 1476395008u; vvv[197] = 2483027968u; vvv[198] = 1040187392u; vvv[199] = 3808428032u; vvv[200] = 3196059648u; vvv[201] = 599785472u; vvv[202] = 505413632u; vvv[203] = 4077912064u; vvv[204] = 1182269440u; vvv[205] = 1736704000u; vvv[206] = 2017853440u; vvv[207] = 2221342720u; vvv[208] = 3329785856u; vvv[209] = 2810494976u; vvv[210] = 3628507136u; vvv[211] = 1416089600u; vvv[212] = 2658719744u; vvv[213] = 864310272u; vvv[214] = 3863387648u; vvv[215] = 3076993792u; vvv[216] = 553150080u; vvv[217] = 272922560u; vvv[218] = 4167467040u; vvv[219] = 1148698640u; vvv[220] = 1719673080u; vvv[221] = 2009075780u; vvv[222] = 2149644390u; vvv[223] = 3222291575u;
    vvv[224] = 2147483648u; vvv[225] = 1073741824u; vvv[226] = 2684354560u; vvv[227] = 1342177280u; vvv[228] = 2281701376u; vvv[229] = 1946157056u; vvv[230] = 436207616u; vvv[231] = 2566914048u; vvv[232] = 2625634304u; vvv[233] = 3208642560u; vvv[234] = 2720006144u; vvv[235] = 2098200576u; vvv[236] = 111673344u; vvv[237] = 2354315264u; vvv[238] = 3464626176u; vvv[239] = 4027383808u; vvv[240] = 2886631424u; vvv[241] = 3770826752u; vvv[242] = 1691164672u; vvv[243] = 3357462528u; vvv[244] = 1993345024u; vvv[245] = 3752330240u; vvv[246] = 873073152u; vvv[247] = 2870150400u; vvv[248] = 1700563072u; vvv[249] = 87021376u; vvv[250] = 1097028000u; vvv[251] = 1222351248u; vvv[252] = 1560027592u; vvv[253] = 2977959924u; vvv[254] = 23268898u; vvv[255] = 437609937u;
}

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Triangle {
    vec3 v0, v1, v2;
    vec3 vn0, vn1, vn2;
    vec3 vt0, vt1, vt2;
};

struct Material {
    vec3 radiance;
    vec3 Kd;			// Diffuse color
    vec3 Ks;			// Specular color
    float Ns, Ni;		// Specular exponent & Optical density
    int map_id;
};

struct Light {
    vec3 v0, v1, v2;
    vec3 vn0, vn1, vn2;
    vec3 radiance;
    float A;
};

struct HitResult {
    bool isHit;
    bool isInside;
    float time;
    vec3 hitPoint;
    vec3 normal;
    vec3 texture;
    vec3 raydir;
    Material material;
};

struct LightSamplingRecord {
    vec3 w;
    float d;
    float pdf;
};

uint seed = uint(
    uint((pos.x * 0.5 + 0.5) * width) * uint(1973) +
    uint((pos.y * 0.5 + 0.5) * height) * uint(9277) +
    uint(frameCounter) * uint(26699)) | uint(1);

uint seed2 = uint(
    uint(width) * uint(1973) +
    uint(height) * uint(9277) +
    uint(frameCounter) * uint(26699)) | uint(1);

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

float rand() {
    return float(wang_hash(seed)) / 4294967296.0;
}

float rand2() {
    return float(wang_hash(seed2)) / 4294967296.0;
}

float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley2d(uint i, uint N) {
    return vec2(float(i) / float(N), radicalInverse_VdC(i));
}
//
vec3 hemisphereSample_uniform(float u, float v) {
    float phi = v * 2.0 * PI;
    float cosTheta = 1.0 - u;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// Generate Binary Gray Code
uint grayCode(uint i) {
    return i ^ (i >> 1);
}

// Generate the i-th sobol number in d-th demension
float sobol(uint d, uint i) {
    uint res = 0u;
    uint offset = d * 32u;
    for (uint j = 0u; bool(i); i >>= 1u, j++) {
        if (bool(i & 1u)) res ^= vvv[j + offset];
    }
    return float(res) * (1.0f / float(0xFFFFFFFFU));
}

// Generate random vector of the b-th bounce in the i-th frame
vec2 sobolVec2(uint i, uint b) {
    float u = sobol(b * 2u, grayCode(i));
    float v = sobol(b * 2u + 1u, grayCode(i));
    return vec2(u, v);
}

vec2 CranleyPattersonRotation(vec2 p) {
    uint pseed = uint(
        uint((pos.x * 0.5 + 0.5) * width) * uint(1973) +
        uint((pos.y * 0.5 + 0.5) * height) * uint(9277) +
        uint(114514 / 1919) * uint(26699)) | uint(1);

    float u = float(wang_hash(pseed)) / 4294967296.0;
    float v = float(wang_hash(pseed)) / 4294967296.0;

    p.x += u;
    if (p.x > 1) p.x -= 1;
    if (p.x < 0) p.x += 1;

    p.y += v;
    if (p.y > 1) p.y -= 1;
    if (p.y < 0) p.y += 1;

    return p;
}

// ----------------------------------------------------------------------------- //

vec3 SampleHemisphere() {
    float z = rand();
    float r = max(0, sqrt(1.0f - z * z));
    float phi = 2.0f * PI * rand();
    return vec3(r * cos(phi), r * sin(phi), z);
}
vec3 SampleHemisphere2(float xi_1, float xi_2) {
    xi_1 = rand(), xi_2 = rand();
    float z = xi_1;
    float r = max(0, sqrt(1.0 - z * z));
    float phi = 2.0 * PI * xi_2;
    return vec3(r * cos(phi), r * sin(phi), z);
}

vec3 toNormalHemisphere(vec3 v, vec3 N) {
    vec3 helper = vec3(1.0f, 0.0f, 0.0f);
    if (abs(N.x) > 0.999f) helper = vec3(0.0f, 0.0f, 1.0f);
    vec3 tangent = normalize(cross(N, helper));
    vec3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

vec3 SampleCosineHemisphere(float xi_1, float xi_2) {

    float r = sqrt(xi_1);
    float theta = xi_2 * 2.0 * PI;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - x * x - y * y);

    return vec3(x, y, z);
}

Triangle getTriangle(int i) {
    int offset = i * SIZE_TRIANGLE;
    Triangle tri;
    
    tri.v0 = texelFetch(Tris, offset + 0).xyz;
    tri.v1 = texelFetch(Tris, offset + 1).xyz;
    tri.v2 = texelFetch(Tris, offset + 2).xyz;

    tri.vn0 = texelFetch(Tris, offset + 3).xyz;
    tri.vn1 = texelFetch(Tris, offset + 4).xyz;
    tri.vn2 = texelFetch(Tris, offset + 5).xyz;

    tri.vt0 = texelFetch(Tris, offset + 6).xyz;
    tri.vt1 = texelFetch(Tris, offset + 7).xyz;
    tri.vt2 = texelFetch(Tris, offset + 8).xyz;

    return tri;
}

Material getMaterial(int i) {
    int offset = i * SIZE_TRIANGLE;
    Material mater;
    mater.radiance = texelFetch(Tris, offset + 9).xyz;
    mater.Kd = texelFetch(Tris, offset + 10).xyz;
    mater.Ks = texelFetch(Tris, offset + 11).xyz;

    vec3 Ns_Ni_mapid = texelFetch(Tris, offset + 12).xyz;
    mater.Ns = Ns_Ni_mapid.x;
    mater.Ni = Ns_Ni_mapid.y;
    mater.map_id = int(Ns_Ni_mapid.z);
    return mater;
}

Light getLight(int i) {
    int offset = i * SIZE_TRIANGLE;
    Light light;
    light.v0 = texelFetch(Lights, offset + 0).xyz;
    light.v1 = texelFetch(Lights, offset + 1).xyz;
    light.v2 = texelFetch(Lights, offset + 2).xyz;
    //light.v0 = vec3(light.v0.xy, light.v0.z-10.0f);
    //light.v1 = vec3(light.v0.xy, light.v0.z - 10.0f);
    //light.v2 = vec3(light.v0.xy, light.v0.z - 10.0f);

    light.vn0 = texelFetch(Lights, offset + 3).xyz;
    light.vn1 = texelFetch(Lights, offset + 4).xyz;
    light.vn2 = texelFetch(Lights, offset + 5).xyz;

    light.radiance = texelFetch(Lights, offset + 9).xyz;
    light.A = length(cross(light.v0 - light.v1, light.v2 - light.v1));
    return light;
}

struct BVHNode {
    int left;           // left chird
    int right;          // right chird
    int n;              // num of triangles
    int index;          // index of triangle
    vec3 AA, BB;        // bounding box
};

BVHNode getBVHNode(int i) {
    BVHNode node;

    int offset = i * SIZE_BVHNODE;
    vec3 childs = texelFetch(Tree, offset + 0).xyz;
    vec3 leafInfo = texelFetch(Tree, offset + 1).xyz;
    node.left = int(childs.x);
    node.right = int(childs.y);
    node.n = int(leafInfo.x);
    node.index = int(leafInfo.y);

    node.AA = texelFetch(Tree, offset + 2).xyz;
    node.BB = texelFetch(Tree, offset + 3).xyz;

    return node;
}
float hitAABB(Ray r, vec3 AA, vec3 BB) {
    vec3 invdir = 1.0 / r.dir;

    vec3 f = (BB - r.origin) * invdir;
    vec3 n = (AA - r.origin) * invdir;

    vec3 tmax = max(f, n);
    vec3 tmin = min(f, n);

    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    float t0 = max(tmin.x, max(tmin.y, tmin.z));

    return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);
}

vec3 getTexture(int i, vec2 tex) {
    switch (i) {
    case 0:
        return texture2D(texture0, tex).rgb;
    case 1:
        return texture2D(texture1, tex).rgb;
    case 2:
        return texture2D(texture2, tex).rgb;
    case 3:
        return texture2D(texture3, tex).rgb;
    case 4:
        return texture2D(texture4, tex).rgb;
    case 5:
        return texture2D(texture5, tex).rgb;
    case 6:
        return texture2D(texture6, tex).rgb;
    case 7:
        return texture2D(texture7, tex).rgb;
    default:
        return texture2D(texture0, tex).rgb;
    }
}

HitResult hitTriangle(Triangle triangle, Ray ray) {
    HitResult res;
    res.time = INF;
    res.isHit = false;
    res.isInside = false;

    vec3 v0 = triangle.v0;
    vec3 v1 = triangle.v1;
    vec3 v2 = triangle.v2;

    vec3 O = ray.origin;
    vec3 D = ray.dir;
    vec3 N = normalize(cross(v1 - v0, v2 - v0));

    if (dot(N, D) > 0.0f) {
        N = -N;
        res.isInside = true;
    }

    if (abs(dot(N, D)) < 0.00001f) return res;


    float t = (dot(N, v0) - dot(O, N)) / dot(D, N);
    if (t < 0.0005f) return res;

    vec3 P = O + D * t;

    // Check the intersection point whether in triangle
    vec3 c1 = cross(v1 - v0, P - v0);
    vec3 c2 = cross(v2 - v1, P - v1);
    vec3 c3 = cross(v0 - v2, P - v2);
    bool r1 = (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0);
    bool r2 = (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0);

    // Hit and return
    if (r1 || r2) {
        res.isHit = true;
        res.hitPoint = P;
        res.time = t;
        res.raydir = D;
        float alpha = (-(P.x - v1.x) * (v2.y -v1.y) + (P.y - v1.y) * (v2.x - v1.x)) / (-(v0.x - v1.x - 0.00005) * (v2.y - v1.y + 0.00005) + (v0.y - v1.y + 0.00005) * (v2.x - v1.x + 0.00005));
        float beta = (-(P.x - v2.x) * (v0.y - v2.y) + (P.y - v2.y) * (v0.x - v2.x)) / (-(v1.x - v2.x - 0.00005) * (v0.y - v2.y + 0.00005) + (v1.y - v2.y + 0.00005) * (v0.x - v2.x + 0.00005));
        float gama = 1.0 - alpha - beta;
        vec3 VN = alpha * triangle.vn0 + beta * triangle.vn1 + gama * triangle.vn2;
        vec3 VT = alpha * triangle.vt0 + beta * triangle.vt1 + gama * triangle.vt2;
        VN = normalize(VN);
        res.normal = (res.isInside) ? (-VN) : (VN);
        res.texture = VT;
    }
    return res;
}

HitResult hitArray(Ray ray, int l, int r) {
    HitResult res;
    res.isHit = false;
    res.time = INF;
    for (int i = l; i <= r; i++) {
        Triangle triangle = getTriangle(i);
        HitResult r = hitTriangle(triangle, ray);
        if (r.isHit && r.time < res.time) {
            res = r;
            res.material = getMaterial(i);
        }
    }
    return res;
}
HitResult hitBVH(Ray ray) {
    HitResult res;
    res.isHit = false;
    res.time = INF;

    int stack[64];
    int sp = 0;

    stack[sp++] = 0;
    while (sp > 0) {
        int top = stack[--sp];
        BVHNode node = getBVHNode(top);

        // Leaf node, then find the nearest point of intersection
        if (node.n > 0) {
            int L = node.index;
            int R = node.index + node.n - 1;
            HitResult r = hitArray(ray, L, R);
            if (r.isHit && r.time < res.time)
                res = r;
            continue;
        }

        // Intersect with the left and the right AABB bounding box
        float d1 = INF; // Distance to the left box
        float d2 = INF; // Distance to the right box
        if (node.left > 0) {
            BVHNode leftNode = getBVHNode(node.left);
            d1 = hitAABB(ray, leftNode.AA, leftNode.BB);
        }
        if (node.right > 0) {
            BVHNode rightNode = getBVHNode(node.right);
            d2 = hitAABB(ray, rightNode.AA, rightNode.BB);
        }

        // Search in the nearest box
        if (d1 > 0 && d2 > 0) {
            if (d1 < d2) {
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            }
            else {
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        }
        else if (d1 > 0) {
            stack[sp++] = node.left;
        }
        else if (d2 > 0) {
            stack[sp++] = node.right;
        }
    }
    return res;
}
bool sameHemisphere(in vec3 n, in vec3 a, in vec3 b) {
    return ((dot(n, a) * dot(n, b)) > 0.0);
}

bool sameHemisphere(in vec3 a, in vec3 b) {
    return (a.z * b.z > 0.0);
}
bool is_inf(float val) {
    return val != val;
    //return isinf(val);	//webGL 2.0 is required
}

float cosTheta(vec3 w) { return w.z; }
float cosTheta2(vec3 w) { return cosTheta(w) * cosTheta(w); }
float absCosTheta(vec3 w) { return abs(w.z); }
float sinTheta2(vec3 w) { return max(0.0, 1.0 - cosTheta2(w)); }
float sinTheta(vec3 w) { return sqrt(sinTheta2(w)); }
float tanTheta2(vec3 w) { return sinTheta2(w) / cosTheta2(w); }
float tanTheta(vec3 w) { return sinTheta(w) / cosTheta(w); }

float cosPhi(vec3 w) { float sin_Theta = sinTheta(w); return (sin_Theta == 0.0) ? 1.0 : clamp(w.x / sin_Theta, -1.0, 1.0); }
float sinPhi(vec3 w) { float sin_Theta = sinTheta(w); return (sin_Theta == 0.0) ? 0.0 : clamp(w.y / sin_Theta, -1.0, 1.0); }
float cosPhi2(vec3 w) { return cosPhi(w) * cosPhi(w); }
float sinPhi2(vec3 w) { return sinPhi(w) * sinPhi(w); }

float ggx_eval(vec3 wh, float alphax, float alphay) {
    float tan2Theta = tanTheta2(wh);
    if (is_inf(tan2Theta)) return 0.;
    float cos4Theta = cosTheta2(wh) * cosTheta2(wh);
    float e = ((cosPhi2(wh) + sinPhi2(wh)) / (alphax * alphay)) * tan2Theta;
    return 1.0 / (PI * (alphax * alphay) * cos4Theta * (1.0 + e) * (1.0 + e));
}

//Here we sample only visible normals, so it takes view direction wi
//Visible normal sampling was first presented here: https://hal.archives-ouvertes.fr/hal-01509746
//We use method which first converts everything is space where alpha is 1 
//does uniform sampling of visible hemisphere and converts sample back
//https://hal.archives-ouvertes.fr/hal-01509746
vec3 ggx_sample(vec3 wi, float alphax, float alphay, vec2 xi) {
    //stretch view
    vec3 v = normalize(vec3(wi.x * alphax, wi.y * alphay, wi.z));

    //orthonormal basis
    vec3 t1 = (v.z < 0.9999) ? normalize(cross(v, vec3(0.0, 0.0, 1.0))) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(t1, v);

    //sample point with polar coordinates
    float a = 1.0 / (1.0 + v.z);
    float r = sqrt(xi.x);
    float phi = (xi.y < a) ? xi.y / a * PI : PI + (xi.y - a) / (1.0 - a) * PI;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi) * ((xi.y < a) ? 1.0 : v.z);

    //compute normal
    vec3 n = p1 * t1 + p2 * t2 + v * sqrt(1.0 - p1 * p1 - p2 * p2);

    //unstretch
    return normalize(vec3(n.x * alphax, n.y * alphay, n.z));
}


float ggx_lambda(vec3 w, float alphax, float alphay) {
    float absTanTheta = abs(tanTheta(w));
    if (is_inf(absTanTheta)) return 0.;
    // Compute _alpha_ for direction _w_
    float alpha_ = sqrt((cosPhi2(w) + sinPhi2(w)) * (alphax * alphay));
    float alpha2Tan2Theta = (alpha_ * absTanTheta) * (alpha_ * absTanTheta);
    return (-1.0 + sqrt(1.0 + alpha2Tan2Theta)) / 2.0;
}

float ggx_g1(vec3 w, float alphax, float alphay) {
    return 1.0 / (1.0 + ggx_lambda(w, alphax, alphay));
}

float ggx_g(vec3 wo, vec3 wi, float alphax, float alphay) {
    return 1.0 / (1.0 + ggx_lambda(wo, alphax, alphay) + ggx_lambda(wi, alphax, alphay));
}

float ggx_pdf(vec3 wi, vec3 wh, float alphax, float alphay) {
    return ggx_eval(wh, alphax, alphay) * ggx_g1(wi, alphax, alphay) * abs(dot(wi, wh)) / abs(wi.z);
}

float smithG_GGX(float NdotV, float alphaG) {
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

float GTR2(float NdotH, float a) {
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

//float SchlickFresnel(float Rs, float u) {
//    float u2 = u * u;
//    return Rs + u2 * u2 * u * (1. - Rs);
//}

float SchlickFresnel(in float Rs, float cosTheta) {
    return Rs + pow(1.0 - cosTheta, 5.) * (1. - Rs);
}

//taken from: https://www.shadertoy.com/view/4sSSW3
void basis(in vec3 n, out vec3 f, out vec3 r) {
    if (n.z < -0.999999) {
        f = vec3(0, -1, 0);
        r = vec3(-1, 0, 0);
    }
    else {
        float a = 1. / (1. + n.z);
        float b = -n.x * n.y * a;
        f = vec3(1. - n.x * n.x * a, b, -n.x);
        r = vec3(b, 1. - n.y * n.y * a, -n.y);
    }
}

mat3 mat3FromNormal(in vec3 n) {
    vec3 x;
    vec3 y;
    basis(n, x, y);
    return mat3(x, y, n);
}

mat3 mat3Inverse(in mat3 m) {
    return mat3(vec3(m[0][0], m[1][0], m[2][0]),
        vec3(m[0][1], m[1][1], m[2][1]),
        vec3(m[0][2], m[1][2], m[2][2]));
}

float RGB2Gray(vec3 rgb) {
    return 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
}

float pdfDiffuse(in vec3 L_local) {
    return INV_PI * L_local.z;
}

float pdfSpecular(in float alphau, in float alphav, in vec3 E_local, in vec3 L_local) {
    vec3 wh = normalize(E_local + L_local);
    return ggx_pdf(E_local, wh, alphau, alphav) / (4.0 * dot(E_local, wh));
}

vec3 l2w(in vec3 localDir, in vec3 normal) {
    vec3 a, b;
    basis(normal, a, b);
    return localDir.x * a + localDir.y * b + localDir.z * normal;
}

vec3 sphericalToCartesian(in float rho, in float phi, in float theta) {
    float sinTheta = sin(theta);
    return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cos(theta)) * rho;
}

vec3 sampleHemisphereCosWeighted(in vec2 xi) {
#ifdef CONCENTRIC_DISK
    vec2 xy = concentricSampleDisk(xi);
    float r2 = xy.x * xy.x + xy.y * xy.y;
    return vec3(xy, sqrt(max(0.0, 1.0 - r2)));
#else
    float theta = acos(sqrt(1.0 - xi.x));
    float phi = 2 * PI * xi.y;
    return sphericalToCartesian(1.0, phi, theta);
#endif
}

vec3 sampleHemisphereCosWeighted(in vec3 n, in vec2 xi) {
    return l2w(sampleHemisphereCosWeighted(xi), n);
}
float PdfAtoW(float aPdfA, float aDist2, float aCosThere) {
    float absCosTheta = abs(aCosThere);
    if (absCosTheta < 0.00001f)
        return 0.0;

    return aPdfA * aDist2 / absCosTheta;
}

//
//float sampleLightSourcePdf(in vec3 x,
//    in vec3 wi,
//    in float d,
//    in float cosAtLight) {
//    float sph_r2 = objects[0].params_[1];
//    vec3 sph_p = toVec3(objects[0].transform_ * vec4(vec3(0.0, 0.0, 0.0), 1.0));
//    float solidangle;
//    vec3 w = sph_p - x;			//direction to light center
//    float dc_2 = dot(w, w);		//squared distance to light center
//    float dc = sqrt(dc_2);		//distance to light center
//
//    if (dc_2 > sph_r2) {
//        float sin_theta_max_2 = clamp(sph_r2 / dc_2, 0.0, 1.0);
//        float cos_theta_max = sqrt(1.0 - sin_theta_max_2);
//        solidangle = TWO_PI * (1.0 - cos_theta_max);
//    }
//    else {
//        solidangle = FOUR_PI;
//    }
//
//    return 1.0 / solidangle;
//}

vec3 SampleGTR2(float xi_1, float xi_2, float alpha) {

    float phi_h = 2.0 * PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0 - xi_2) / (1.0 + (alpha * alpha - 1.0) * xi_2));
    float sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));
    
    vec3 H = vec3(sin_theta_h * cos_phi_h, sin_theta_h * sin_phi_h, cos_theta_h);

    return H;
}

vec3 mtlEval(int type, Material mtl, in vec3 Ng, in vec3 Ns, in vec3 E, in vec3 L) {
    mat3 trans = mat3FromNormal(Ns);
    mat3 inv_trans = mat3Inverse(trans);

    vec3 E_local = inv_trans * E;
    vec3 L_local = inv_trans * L;
    
    if (type == -1) {
        return vec3(3.0);
    } 
    
    //vec3 N_local = vec3(0.0f, 0.0f, 1.0f);
    vec3 H_local = L_local + E_local;
    H_local = normalize(H_local);
    float EdotH = dot(E_local, H_local);
    float LdotH = dot(L_local, H_local);

    float alpha = 1 / sqrt(mtl.Ns);
    float Roughness = alpha;

    if (!sameHemisphere(E_local, L_local)) {
        return vec3(0.0);
    }

    float cosThetaO = abs(E_local.z), cosThetaI = abs(L_local.z);

    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0.0 || cosThetaO == 0.0) return vec3(0.);
    if (H_local.x == 0.0 && H_local.y == 0.0 && H_local.z == 0.0) return vec3(0.);

    //vec2 kdks = vec2(RGB2Gray(mtl.Kd), RGB2Gray(mtl.Ks)) / vec2(RGB2Gray(mtl.Kd) + RGB2Gray(mtl.Ks) + 0.00001f);

    // Fresnel Coefficient
    float F0 = 0.0;
    // Metallic
    if (mtl.Ns > 1.0) {
        F0 = 0.9;
    }

    float Cdlum = RGB2Gray(mtl.Kd);
    vec3 Ctint = Cdlum > 0 ? mtl.Kd / Cdlum : vec3(1.0);
    vec3 Cspec = mtl.Ks * mix(vec3(1), Ctint, 0.5);
    vec3 Cspec0 = mix(0.08 * Cspec, mtl.Kd, F0);
    // Diffuse

    float Fd90 = 0.5 + 2.0 * EdotH * EdotH * Roughness;
    float FV = SchlickFresnel(F0, cosThetaO);
    float FL = SchlickFresnel(F0, cosThetaI);
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);
    //Fd = 1.;
    //vec3 diff_refl = vec3(INV_PI) * mtl.Kd *(vec3(1.0) - F);
    vec3 diff_refl = INV_PI * Fd * mtl.Kd;

    // Specular
    //vec3 spec_Refl = mtl.Ks * ggx_eval(wh, alpha, alpha) * ggx_g(E_local, L_local, alpha, alpha) * F / (4.0 * cosThetaI * cosThetaO);
    // spec_Refl = mtl.Ks * (mtl.Ns + 2) * pow(H_local.z, mtl.Ns) * vec3(INV_PI) / 2.0;
    //vec3 spec_Refl = vec3(0.0);
    float Ds = GTR2(LdotH, alpha);
    float FH = SchlickFresnel(F0, EdotH);
    vec3 Fs = mix(Cspec0, vec3(1), FH);
    float Gs = smithG_GGX(cosThetaO, Roughness);
    Gs *= smithG_GGX(cosThetaI, Roughness);

    //vec3 spec_Refl = Gs * Fs * Ds;
    vec3 spec_Refl = mtl.Ks;

    return (type == 0) ? (spec_Refl) : ((type == 1) ? diff_refl: mix(diff_refl, spec_Refl, F0));

    //return 	mix(diff_refl, spec_Refl, F0);
}

vec3 mtlSample(bool inside, Material mtl, in vec3 Ng, in vec3 Ns, in vec3 E, in vec2 xi, out vec3 L, out float pdf) {
    float alpha = 1/ sqrt(mtl.Ns);

    mat3 trans = mat3FromNormal(Ns);
    mat3 inv_trans = mat3Inverse(trans);

    //convert directions to local space
    vec3 E_local = inv_trans * E;
    vec3 L_local;

    if (E_local.z == 0.0) return vec3(0.0f);
    
    float F0 = 0.0;
    // Metallic
    if (mtl.Ns > 1.0) {
        F0 = 0.9;
    }
    int type;

    //Sample specular or diffuse lobe based on fresnel
    //vec2 kdks = vec2(dot(mtl.Kd, mtl.Kd), dot(mtl.Ks, mtl.Ks)) / vec2(dot(mtl.Kd, mtl.Kd) + dot(mtl.Ks, mtl.Ks) + 0.00001f);
    //vec2 kdks = vec2(length(mtl.Kd), length(mtl.Ks)) / vec2(length(mtl.Kd) + length(mtl.Ks) + 0.00001f);
    float pdf1, pdf2;
    if (mtl.Ni > 1.0) {
        type = -1;
        float eta = 1.5;
        float sR0 = (mtl.Ni - 1.0) / (mtl.Ni + 1.0);
        float R0 = sR0 * sR0;
        float Ff = SchlickFresnel(R0, abs(E_local.z));
        //vec3 wh = SampleGTR2(xi.x, xi.y, 0.00001);

        if (rand() > Ff) {
            if (!inside) eta = 1.0 / eta;
            L_local = refract(-E_local, vec3(0.0,0.0,1.0), eta);
            L = trans * L_local;
            pdf = 1;
            return mtlEval(type, mtl, Ng, Ns, E, L);
        }
        else {
            L_local = reflect(-E_local, vec3(0.0, 0.0, 1.0));
            vec3 H = normalize(L_local + E_local);
            L = trans * L_local;
            pdf = 1;
            return mtlEval(type, mtl, Ng, Ns, E, L);
        }
        L = trans * L_local;
        pdf = 1.0 / 2 * PI;
        
    }
    else if (rand() < F0) {
        type = 0;
        //vec3 wh = ggx_sample2(E_local, alpha, alpha, xi);
        vec3 wh = SampleGTR2(xi.x, xi.y, alpha);
        L_local = reflect(-E_local, wh);
        vec3 H = normalize(L_local + E_local);
        float ds = GTR2(H.z, alpha);
        //pdf = pdfSpecular(alpha, alpha, E_local, L_local);
        //pdf = ds * H.z / (4.0 * dot(E_local, H));
        pdf = 1;
        L = trans * L_local;
    }
    else {
        type = 1;
        //L_local = sampleHemisphereCosWeighted(xi);
        L_local = SampleCosineHemisphere(xi.x, xi.y);
        pdf = pdfDiffuse(L_local);
        L = trans * L_local;
    }
    //L = trans * L_local;
    if (!sameHemisphere(Ns, E, L) || !sameHemisphere(Ng, E, L)) {
        pdf = 0.0;
    }
    //return RED;
    //L_local = reflect(-E_local, vec3(0.0, 0.0, 1.0));
    //L = trans * L_local;
    //pdf = 1;
    
    //pdf = ds * H.z / (4.0 * dot(E_local, H)) + pdfDiffuse(L_local);
    //pdf = (pdf1 + pdf2) / 2.0;
    //convert directions to global space
    //pdf = 1 / (2 * PI);
    return mtlEval(type, mtl, Ng, Ns, E, L);
}

float mtlPdf(Material mtl, in vec3 Ng, in vec3 Ns, in vec3 E, in vec3 L) {
    mat3 trans = mat3FromNormal(Ns);
    mat3 inv_trans = mat3Inverse(trans);
    float alpha = 1 / sqrt(mtl.Ns);

    vec3 E_local = inv_trans * E;
    vec3 L_local = inv_trans * L;

    vec2 kdks = vec2(length(mtl.Kd), length(mtl.Ks)) / vec2(length(mtl.Kd) + length(mtl.Ks) + 0.00001f);

    if (!sameHemisphere(Ng, E_local, L_local)) {
        return 0.0;
    }
    float diff_pdf = abs(L_local.z) * INV_PI;

    if (!sameHemisphere(E_local, L_local)) return 0.0;
    vec3 wh = normalize(E_local + L_local);
    float spec_pdf = ggx_pdf(E_local, wh, alpha, alpha) / (4.0 * dot(E_local, wh));

    return mix(diff_pdf, spec_pdf, kdks.y);
}

vec3 sampleBSDF(in HitResult hit,
                in bool useMIS,
                in int strataCount,
                in int strataIndex,
                out vec3 wo,
                out float brdfPdfW,
                out vec3 fr,
                out HitResult newHit,
                out bool ishit) {
    vec3 Lo = vec3(0.0);
    vec3 ns = hit.normal, ng = hit.normal;
    Material mtl = hit.material;
    vec3 wi = -hit.raydir;
    bool inside = hit.isInside;

    for (int i = 0; i < DL_SAMPLES; i++) {
        vec2 xi = sobolVec2(uint(strataCount) + uint(i), uint(strataIndex));
        xi = CranleyPattersonRotation(xi);
        fr = mtlSample(inside ,mtl, ng, ns, wi, xi, wo, brdfPdfW);
        
        //fr = eval(mtl, ng, ns, wi, wo);
        float dotNWo = dot(wo, ns);
        //if (dotNWo == 0.0) return RED;
        //Continue if sampled direction is under surface
        if ((dot(fr, fr) > 0.0) && (brdfPdfW > 0.00001f)) {
            Ray shadowRay;
            shadowRay.origin = hit.hitPoint;
            shadowRay.dir = wo;
            //abstractLight* pLight = 0;
            float cosAtLight = 1.0;
            float distanceToLight = -1.0;
            vec3 Li = vec3(0.0);

            {
                //float distToHit;

                newHit = hitBVH(shadowRay);

                //newHit = newhit;
                
                if (newHit.isHit) {
                    if (length(newHit.material.radiance) > 0.0001f) {
                        distanceToLight = newHit.time;
                        cosAtLight = dot(newHit.normal, -wo);
                        //if (cosAtLight > 0.0 && newHit.isInside) {
                        if (cosAtLight > 0.0 && !newHit.isInside) {
                            Li = newHit.material.radiance;
                            ishit = false;
                        }
                    }
                    else{
                        ishit = true;
                    }
                }
                else {
                    ishit = false;
                    //TODO check for infinite lights
                }
            }
            
            if (distanceToLight > 0.0) {
                if (cosAtLight > 0.0) {
                    
                    vec3 contribution = (Li * fr * abs(dotNWo)) / (brdfPdfW+0.0001f);
                    //return vec3(1.0, 0.0, 0.0);

                    //if (useMIS/* && !(mtl->isSingular())*/) {
                    //    float lightPickPdf = 1.0;//lightPickingPdf(x, n);
                    //    float lightPdfW = sampleLightSourcePdf(newHit.hitPoint, wi, distanceToLight, cosAtLight);
                    //    //float lightPdfW = sphericalLightSamplingPdf( x, wi );//pLight->pdfIlluminate(x, wo, distanceToLight, cosAtLight) * lightPickPdf;

                    //    contribution *= misWeight(brdfPdfW, lightPdfW);
                    //}
                    Lo += contribution;
                }
            }
        }
    }
    return Lo / float(DL_SAMPLES);
}

bool isLightVisible(Ray shadowRay, float lightDist) {
    HitResult tmpHit;
    tmpHit = hitBVH(shadowRay);

    return tmpHit.isHit && (tmpHit.time > lightDist - 0.0001f) && (tmpHit.time < lightDist + 0.0001f);
}

vec3 sampleLightSource(in vec3 x, Light lit,
    float Xi1, float Xi2,
    out LightSamplingRecord sampleRec) { //sampleRec.w .pdf .d
    float Xi3 = 1.0 - Xi1 - Xi2;
    vec3 hitPoint = Xi1 * lit.v0 + Xi2 * lit.v1 + Xi3 * lit.v2;
    vec3 dir = normalize(hitPoint - x);
    float dist = distance(hitPoint, x);
    float dist2 = dist * dist;
    float alpha = (-(hitPoint.x - lit.v1.x) * (lit.v2.y - lit.v1.y) + (hitPoint.y - lit.v1.y) * (lit.v2.x - lit.v1.x)) / (-(lit.v0.x - lit.v1.x - 0.00005) * (lit.v2.y - lit.v1.y + 0.00005) + (lit.v0.y - lit.v1.y + 0.00005) * (lit.v2.x - lit.v1.x + 0.00005));
    float beta = (-(hitPoint.x - lit.v2.x) * (lit.v0.y - lit.v2.y) + (hitPoint.y - lit.v2.y) * (lit.v0.x - lit.v2.x)) / (-(lit.v1.x - lit.v2.x - 0.00005) * (lit.v0.y - lit.v2.y + 0.00005) + (lit.v1.y - lit.v2.y + 0.00005) * (lit.v0.x - lit.v2.x + 0.00005));
    float gamma = 1.0f - alpha - beta;
    vec3 VN = alpha * lit.vn0 + beta * lit.vn1 + gamma * lit.vn2;
    float acos = dot(-dir, normalize(VN));
    if (acos <= 0.0) return vec3(0.0);
    sampleRec.w = dir;
    sampleRec.d = dist;
    sampleRec.pdf = PdfAtoW(1.0 / (lit.A + 0.0001f), dist2, acos);

    return lit.radiance;
}

vec3 salmpleLight(in HitResult hit,
                    in bool useMIS,
                    in int strataCount,
                    in int strataIndex) {
    vec3 Lo = vec3(0.0);	//outgoing radiance
    
    for (int i = 0; i < DL_SAMPLES; i++) {
        //for (int j = 0; j < nlights; ++j) {
        vec3 Loj = vec3(0.0);
        int count = 4;
        int COUNT = count;
        //return vec3(0.0, 1.0, 0.0);
        for (int j = 0; j < nlights; ++j) {
            if (nlights > 10) {
                j = int(rand() * nlights);
                j--;
            }
            float lightPickingPdf = 1.0f;
            //Light light = pickOneLight(lightPickingPdf);
            vec3 ns = hit.normal, ng = hit.normal;
            Material mtl = hit.material;
            vec3 wo;
            vec3 wi = -hit.raydir;
            float lightPdfW, lightDist;

            LightSamplingRecord rec;
            /*float Xi1 = rnd();
            float Xi2 = rnd();
            float strataSize = 1.0 / float(strataCount);
            Xi2 = strataSize * (float(strataIndex) + Xi2);*/

            vec2 uv = sobolVec2(uint(strataCount), uint(strataIndex)+uint(j));
            Light lit = getLight(j);
            
            
            vec3 Li = sampleLightSource(hit.hitPoint, lit, uv.x, uv.y, rec);
            
            //vec3 Li = sampleSphericalLight( x, Xi1, Xi2, rec );
            wo = rec.w;
            lightPdfW = rec.pdf;
            lightDist = rec.d;
            lightPdfW *= lightPickingPdf;
            
            float dotNWo = dot(wo, ns);
            
            if ((dot(wo, ng) > 0.0) && (dotNWo > 0.0) && (lightPdfW > 0.00001f)) {
                vec3 fr = mtlEval(1, mtl, ng, ns, wi, wo);
                if (dot(fr, fr) > 0.0) {
                    Ray shadowRay;
                    shadowRay.origin = hit.hitPoint;
                    shadowRay.dir = wo;
                    if (isLightVisible(shadowRay, lightDist)) {
                        vec3 contribution = (Li * fr * dotNWo) / (lightPdfW+0.000001f);

                        //if (useMIS /*&& !(light->isSingular())*/) {
                        //    float brdfPdfW = mtlPdf(mtl, ng, ns, wi, wo);
                        //    contribution *= misWeight(lightPdfW, brdfPdfW);
                        //}

                        Loj += contribution;
                    }
                }
            }
            if (nlights > 10) {
                if (--count == 0) {
                    Loj *= float(nlights) / float(COUNT) /4.0;
                    break;
                } 
            }
            
        }
        Lo += Loj;
    }
    return Lo / float(DL_SAMPLES);
}

vec3 pathTracing(Ray ray, int maxBounce, float p) {

    vec3 Lo = vec3(0.0f);      // ���յ���ɫ
    vec3 history = vec3(1.0f); // �ݹ���۵���ɫ
    HitResult hit = hitBVH(ray);

    if (!hit.isHit) {
        return vec3(0.0f);
    }
    else {
        if (length(hit.material.radiance) > 0.0001f) {
            float cosAtLight = dot(hit.normal, -ray.dir);
            if (cosAtLight > 0.0 && !hit.isInside) {
                Lo = hit.material.radiance;
                return Lo;
            }
        }
    }
    
    for (int bounce = 0; bounce < maxBounce; bounce++) {
        if (length(hit.material.radiance) > 0.0001f || rand() > p) break;


        if (hit.material.map_id != -1) {
            int map_id = hit.material.map_id;
            vec2 tex = vec2(hit.texture.x - floor(hit.texture.x), hit.texture.y - floor(hit.texture.y));
            vec3 baseColor = getTexture(map_id, tex);
            hit.material.Kd = pow(baseColor, vec3(2.2f));
            hit.material.map_id = -1;
        }

        HitResult newHit;
        
        vec3 wo, fr;
        float woPdf;
        bool ishit = false;
        //return vec3(0.0, 0.0, 1.0);
        bool test;
        vec3 directLight;
        bool oo = false;
        if (nlights > 10 && bounce != 0 && oo) {
            directLight = vec3(0.0);
        }
        else {
            directLight = salmpleLight(hit, test, frameCounter + 1, bounce);
        }
        
        directLight += sampleBSDF(hit, false, frameCounter + 2, bounce, wo, woPdf, fr, newHit, ishit);;
        //return directLight;
        float dotWoN = dot(hit.normal, wo);
        Lo += directLight * history;

        if (!ishit) { break; }

        hit = newHit;

        history *= fr * abs(dotWoN) / (woPdf + 0.00001f);
    }
    return Lo / p;
}

void main()
{
    init_vvv();

    Ray ray;
    //vec2 uv = sobolVec2(uint(frameCounter+1), 0u);
    vec2 uv = vec2(rand(), rand());
    ray.origin = eye;
    vec2 scr = vec2(pos.x * float(width) / float(height), pos.y);
    vec2 Point = vec2((uv.x - 0.5f)/ float(width), (uv.y - 0.5f) / float(height));
    float disZ = 1.0f / tan(radians(fovy / 2.0f));
    vec4 dir = view * vec4(scr.xy + Point, -disZ, 0.0);
    ray.dir = normalize(dir.xyz);
    vec3 color;
    
    color = pathTracing(ray, DEPTH, 0.8);

    vec3 last = texture2D(lastframe, 0.5f * pos.xy + 0.5f).rgb;

    if (frameCounter >= 1) {
        color = mix(last, color, 1.0f / float(frameCounter + 1));
    }
    update = vec4(color, 1.0f);
}