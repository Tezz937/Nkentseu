#include <Unitest/Unitest.h>
#include <Unitest/TestMacro.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <iostream>
#include <random>
#include <cstdlib>

#include "NKLogger/NkLog.h"
#include "NKMath/NKMath.h"
#include "SVD.h"
#include "Quat.h" 
#include "IntegralImage.h" 

using namespace NkMath;

// Dimensions de l'image de rendu
const int width = 640, height = 640;
NkImage img(width, height);

// Générateur aléatoire avec graine fixe pour reproductibilité
std::mt19937 rng(137);
std::uniform_real_distribution<double> dist(-10.0, 10.0);

// Sommets du cube unitaire centré en l'origine
std::vector<Vec4d> cube = {
    {-0.5,-0.5,-0.5,1}, {0.5,-0.5,-0.5,1},
    {0.5, 0.5,-0.5,1}, {-0.5, 0.5,-0.5,1},
    {-0.5,-0.5, 0.5,1}, {0.5,-0.5, 0.5,1},
    {0.5, 0.5, 0.5,1}, {-0.5, 0.5, 0.5,1}
};

// Les 12 arêtes du cube définies par paires d'indices
std::vector<Vec2d> edges = {
    {0,1},{1,2},{2,3},{3,0}, // face arrière
    {4,5},{5,6},{6,7},{7,4}, // face avant
    {0,4},{1,5},{2,6},{3,7}  // arêtes latérales
};

// Paramètres de la caméra
Vec3d eye{0,2,4}, target{0,0,0}, up{0,1,0};
Mat4d V = LookAt(eye, target, up);
Mat4d P = Perspective(45.0, double(width)/height, 0.1, 100.0);


// ============================================================
// TP1 : Analyse de la représentation IEEE 754 d'un flottant
// ============================================================
TEST_CASE(Loic_TP1, AnalyseRepresentationIEEE754) {
    inspectFloat(0.5f);
    inspectFloat(2.0f);
    inspectFloat(1.0f / 0.0f);
    inspectFloat(std::sqrt(-1.0f));
    inspectFloat(-0.0f);
    inspectFloat(0.0f);
    inspectFloat(std::numeric_limits<float>::min());
}

// ============================================================
// TP2 : Comparaison des méthodes de sommation et de variance
// ============================================================
TEST_CASE(Loic_TP2, PrecisionNumerique) {
    float s1, s2;
    std::vector<float> v;

    std::vector<float> data(1'000'000, 0.1f);
    s1 = std::accumulate(data.begin(), data.end(), 0.0f);
    s2 = kahanSum(data);
    logger.Info("\nSomme accumulate : {0}\nSomme Kahan      : {1}\nValeur reelle    : 100000.0", s1, s2);
    
    v = std::vector<float>({1e7f, 1e7f, 2.0f, 3.0f});
    logger.Info("\nVariance Naive   : {0}\nVariance Welford : {1}", varianceNaive(v), varianceWelford(v));
    
    logger.Info("\nEpsilon (boucle) : {0}\nEpsilon (std)    : {1}", epsilonMachine(), std::numeric_limits<float>::epsilon());
}

// ============================================================
// TP3 : 33 tests unitaires sur Float.h
// ============================================================
TEST_CASE(Loic_TP3, ValidationFonctionsFloat) {
    // --- isFiniteValid (5 tests) ---
    ASSERT_TRUE(!isFiniteValid(std::numeric_limits<float>::quiet_NaN())); // 1
    ASSERT_TRUE(!isFiniteValid(std::numeric_limits<float>::infinity()));  // 2
    ASSERT_TRUE(!isFiniteValid(-std::numeric_limits<float>::infinity())); // 3
    ASSERT_TRUE(isFiniteValid(0.0f));                                     // 4
    ASSERT_TRUE(isFiniteValid(42.0f));                                    // 5

    // --- nearlyZero (8 tests) ---
    ASSERT_TRUE(nearlyZero(0.0f, 1e-6f));     // 6
    ASSERT_TRUE(!nearlyZero(1e-5f, 1e-6f));   // 7
    ASSERT_TRUE(nearlyZero(1e-7f, 1e-6f));    // 8
    ASSERT_TRUE(nearlyZero(-1e-7f, 1e-6f));   // 9
    ASSERT_TRUE(!nearlyZero(-1e-6f, 1e-7f));  // 10
    ASSERT_TRUE(nearlyZero(1e-3f, 1e-2f));    // 11
    ASSERT_TRUE(!nearlyZero(1e-2f, 1e-3f));   // 12
    ASSERT_TRUE(nearlyZero(5e-8f, 1e-7f));    // 13

    // --- approxEq (10 tests) ---
    ASSERT_TRUE(approxEq(1.0f, 1.0f, 1e-6f));           // 14
    ASSERT_TRUE(approxEq(1.0f, 1.0000001f, 1e-5f));     // 15
    ASSERT_TRUE(!approxEq(1.0f, 1.1f, 1e-3f));          // 16
    ASSERT_TRUE(approxEq(0.0f, 1e-7f, 1e-6f));          // 17
    ASSERT_TRUE(!approxEq(0.0f, 1e-4f, 1e-6f));         // 18
    ASSERT_TRUE(approxEq(-1.0f, -1.000001f, 1e-5f));    // 19
    ASSERT_TRUE(!approxEq(-1.0f, -1.1f, 1e-2f));        // 20
    ASSERT_TRUE(approxEq(500.0f, 500.0001f, 1e-3f));    // 21
    ASSERT_TRUE(approxEq(1200.0f, 1201.0f, 1e-3f));       // 22
    ASSERT_TRUE(approxEq(1e-7f, 2e-7f, 1e-6f));         // 23

    // --- kahanSum vs accumulate (10 tests) ---
    float s1, s2;
    std::vector<float> v;

    v = std::vector<float>(1000, 0.1f);
    s1 = std::accumulate(v.begin(), v.end(), 0.0f);
    s2 = kahanSum(v);
    ASSERT_TRUE(std::fabs(s2 - 100.0f) < std::fabs(s1 - 100.0f)); // 24
    
    v = std::vector<float>(10000, 0.1f);
    s1 = std::accumulate(v.begin(), v.end(), 0.0f);
    s2 = kahanSum(v);
    ASSERT_TRUE(std::fabs(s2 - 1000.0f) < std::fabs(s1 - 1000.0f)); // 25
    
    v = std::vector<float>({1e8f, 1.0f, -1e8f});
    s1 = std::accumulate(v.begin(), v.end(), 0.0f);
    s2 = kahanSum(v);
    ASSERT_TRUE(std::fabs(s2 - 1.0f) <= std::fabs(s1 - 1.0f)); // 26

    v = std::vector<float>({1.0f, 1e8f, -1e8f});
    s1 = std::accumulate(v.begin(), v.end(), 0.0f);
    s2 = kahanSum(v);
    ASSERT_TRUE(std::fabs(s2 - 1.0f) <= std::fabs(s1 - 1.0f)); // 27
    
    v = std::vector<float>(100000, 0.01f);
    s2 = kahanSum(v);
    ASSERT_TRUE(approxEq(s2, 1000.0f, 1e-2f)); // 28

    v = std::vector<float>(100000, 1e-5f);
    s2 = kahanSum(v);
    ASSERT_TRUE(approxEq(s2, 1.0f, 1e-3f)); // 29
    
    v = std::vector<float>({0.2f, 0.3f, 0.5f});
    s2 = kahanSum(v);
    ASSERT_TRUE(approxEq(s2, 1.0f, 1e-6f)); // 30

    v = std::vector<float>(50000, 0.2f);
    s2 = kahanSum(v);
    ASSERT_TRUE(approxEq(s2, 10000.0f, 1e-2f)); // 31
    
    v = std::vector<float>({1e7f, 1.0f, 1.0f, -1e7f});
    s2 = kahanSum(v);
    ASSERT_TRUE(approxEq(s2, 2.0f, 1e-3f)); // 32

    v = std::vector<float>(1000000, 0.1f);
    s2 = kahanSum(v);
    ASSERT_TRUE(approxEq(s2, 100000.0f, 1e-1f)); // 33
}

// ============================================================
// TP4 : Vec2d — produit scalaire, vectoriel 2D, normalisation
// ============================================================
TEST_CASE(Loic_TP4, Vec2dOperations) {
    ASSERT_TRUE(Dot({1,0}, {0,1}) == 0.0);      // 1
    ASSERT_TRUE(Dot({1,0}, {1,0}) == 1.0);      // 2
    ASSERT_TRUE(Dot({3,4}, {3,4}) == 25.0);     // 3
    ASSERT_TRUE(Dot({-1,0}, {1,0}) == -1.0);    // 4
    ASSERT_TRUE(Dot({2,3}, {4,5}) == 23.0);     // 5
    ASSERT_TRUE(Dot({0,0}, {5,7}) == 0.0);      // 6
    
    ASSERT_TRUE(Cross2D({1,0}, {0,1}) == 1.0);   // 7
    ASSERT_TRUE(Cross2D({0,1}, {1,0}) == -1.0);  // 8
    ASSERT_TRUE(Cross2D({1,1}, {1,1}) == 0.0);   // 9
    ASSERT_TRUE(Cross2D({2,0}, {0,2}) == 4.0);   // 10
    
    Vec2d w = {3,4};
    Vec2d n = w.Normalized();
    ASSERT_TRUE(std::fabs(n.Norm() - 1.0) < kEps);   // 11
    ASSERT_TRUE(std::fabs(n.x - 0.6) < kEps);        // 12
    ASSERT_TRUE(std::fabs(n.y - 0.8) < kEps);        // 13
    Vec2d u = {1,0};
    u = u.Normalized();
    ASSERT_TRUE(std::fabs(u.x - 1.0) < kEps);        // 14
    
    w = {10, 20};
    ASSERT_TRUE(w[0] == 10.0);   // 15
    ASSERT_TRUE(w[1] == 20.0);   // 16
    w[0] = 30;
    ASSERT_TRUE(w.x == 30.0);    // 17
    w[1] = 40;
    ASSERT_TRUE(w.y == 40.0);    // 18
    u = {5, 6};
    ASSERT_TRUE(u[0] == 5.0);    // 19
    
    static_assert(sizeof(Vec2d) == 16, "Vec2d doit faire 16 octets"); // 20
}

// ============================================================
// TP5 : Vec3d — produit vectoriel et Gram-Schmidt
// ============================================================
TEST_CASE(Loic_TP5, Vec3dEtGramSchmidt) {
    Vec3d i = {1,0,0}, j = {0,1,0}, k = {0,0,1};

    ASSERT_TRUE(ApproxVec(Cross(i, j), k));               // 1
    ASSERT_TRUE(ApproxVec(Cross(j, i), {0,0,-1}));        // 2
    ASSERT_TRUE(ApproxVec(Cross(j, k), i));               // 3
    ASSERT_TRUE(ApproxVec(Cross(k, i), j));               // 4
    ASSERT_TRUE(approxEq(Dot(Cross(i, j), i), 0));        // 5
    ASSERT_TRUE(approxEq(Dot(Cross(i, j), j), 0));        // 6
    
    for(int t = 0; t < 10; ++t) {
        Vec3d a{dist(rng), dist(rng), dist(rng)};
        Vec3d b{dist(rng), dist(rng), dist(rng)};
        Vec3d c{dist(rng), dist(rng), dist(rng)};
    
        Vec3d ui = a.Normalized();
        Vec3d vi = (b - Project(b, ui)).Normalized();
        Vec3d wi = (c - Project(c, ui) - Project(c, vi)).Normalized();

        ASSERT_TRUE(approxEq(ui.Norm(), 1.0));  // 7
        ASSERT_TRUE(approxEq(vi.Norm(), 1.0));  // 8
        ASSERT_TRUE(approxEq(wi.Norm(), 1.0));  // 9
        ASSERT_TRUE(approxEq(Dot(ui, vi), 0.0));  // 10
        ASSERT_TRUE(approxEq(Dot(ui, wi), 0.0));  // 11
        ASSERT_TRUE(approxEq(Dot(vi, wi), 0.0));  // 12
    }
    
    i = {3,4,0}; j = {1,0,0};
    Vec3d proj = Project(i, j);
    Vec3d rej  = Reject(i, j);
    ASSERT_TRUE(ApproxVec(proj + rej, i)); // 13
}

// ============================================================
// TP6 : Vec4d et projection perspective d'un cube
// ============================================================
TEST_CASE(Loic_TP6, Vec4dProjectionPerspective) {
    std::vector<Vec2d> proj;
    
    double z_cam = 3.0;
    for(auto& p : cube){
        p.z += z_cam;
        proj.push_back(ProjectPoint(p));
    }
    
    for(const auto& p : proj) {
        int x = (int)p.x, y = (int)p.y;
        for(int dx = -2; dx <= 2; dx++)
            for(int dy = -2; dy <= 2; dy++)
                img.SetPixelRGBA(x+dx, y+dy, 255, 0, 0);
    }
    
    for(auto edge : edges)
        img.DrawLine((int)proj[edge.x].x, (int)proj[edge.x].y,
                     (int)proj[edge.y].x, (int)proj[edge.y].y);
    img.SavePPM("cube_TP6.ppm");
}

// ============================================================
// TP7 : Mat4d — identité, inversion, rotation
// ============================================================
TEST_CASE(Loic_TP7, Mat4dInverseEtRotation) {
    Mat4d m, r, inv;

    for(int t = 0; t < 10; t++) {
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++)
                m(i, j) = dist(rng);
        
        r = m * Mat4d::Identity();
        ASSERT_TRUE(ApproxMat(r, m));
    
        if(Inverse(m, inv))
            ASSERT_TRUE(ApproxMat(m * inv, Mat4d::Identity(), 1e-10f));
    }
    
    m = Mat4d::Identity();
    for(int j = 0; j < 4; j++)
        m(1, j) = m(0, j);
    ASSERT_TRUE(!Inverse(m, inv));
    
    r = Mat4d::RotateAxis({0,1,0}, NKENTSEU_PI_DOUBLE / 2.0f);
    Vec4d s = {1,0,0,1}, q = r * s;
    ASSERT_TRUE(approxEq(q.x,  0.0));
    ASSERT_TRUE(approxEq(q.y,  0.0));
    ASSERT_TRUE(approxEq(q.z, -1.0));
}

// ============================================================
// TP8 : Rasteriseur logiciel — cube en rotation
// ============================================================
TEST_CASE(Loic_TP8, RasteriseurCubeEnRotation) {
    for(int frame = 0; frame < 10; frame++){
        img.Fill(255, 255, 255);
        double angle = frame * 0.25;
        Mat4d R = Mat4d::RotateAxis(up, angle);
        std::vector<Vec3d> screen;
    
        for(auto v : cube){
            Vec4d p = P * (V * (R * v));
            screen.push_back(ProjectToScreen(p, width, height));
        }
    
        for(auto edge : edges)
            img.DrawLine((int)screen[edge.x].x, (int)screen[edge.x].y,
                         (int)screen[edge.y].x, (int)screen[edge.y].y, 255);
        img.SavePPM("frame_TP8_"+std::to_string(frame)+".ppm");
    }
}

// ============================================================
// TP9 : TRS — construction et décomposition
// ============================================================
TEST_CASE(Loic_TP9, TRSEtDecomposition) {
    dist = std::uniform_real_distribution<double>(-5.0, 5.0);

    for(int t = 0; t < 20; t++){
        Vec3d outT{dist(rng), dist(rng), dist(rng)};
        Vec3d outR{dist(rng), dist(rng), dist(rng)};
        Vec3d outS{dist(rng) + 6, dist(rng) + 6, dist(rng) + 6};
        
        Mat4d M = TRS(outT, outR, outS);
        Vec3d T2, R2, S2;
        DecomposeTRS(M, T2, R2, S2);
    
        ASSERT_TRUE(ApproxVec(outT, T2));
        ASSERT_TRUE(ApproxVec(outS, S2));
        ASSERT_TRUE(ApproxVec(outR, R2, 5.0));
    }
}

// ============================================================
// TP10 : Quaternions — rotation, conversion Mat3d, inverse
// ============================================================
TEST_CASE(Loic_TP10, Quaternions) {
    Mat3d m1, m2, m3;
    Quat q1, q2, q3;

    Vec3d i = {1,0,0};
    q1 = FromAxisAngle({0,1,0}, NKENTSEU_PI_DOUBLE / 2.0f);
    Vec3d j = Rotate(q1, i);

    ASSERT_TRUE(std::fabs(j.x - 0.0) < kEps);
    ASSERT_TRUE(std::fabs(j.y - 0.0) < kEps);
    ASSERT_TRUE(std::fabs(j.z + 1.0) < kEps);
    
    dist = std::uniform_real_distribution<double>(-1.0, 1.0);
    for(int t = 0; t < 50; t++){
        q1 = { dist(rng), dist(rng), dist(rng), dist(rng) };
        q1 = q1.Normalized();
        m1 = ToMat3(q1);
        q2 = FromMat3(m1);
        q2 = q2.Normalized();
        ASSERT_TRUE(ApproxQuat(q1, q2, 1e-4f));
    }
    
    for(int t = 0; t < 50; t++){
        q1 = { dist(rng), dist(rng), dist(rng), dist(rng) };
        q1 = q1.Normalized();
        q2 = q1.Inverse();
        q3 = q1 * q2;
        ASSERT_TRUE(ApproxQuat(q3, Quat::Identity(), 1e-4f));
    }
}

// ============================================================
// TP11 : Animation SLERP et LERP sur 60 frames
// ============================================================
TEST_CASE(Loic_TP11, AnimationSlerpEtLerp) {
    Quat q1, q2;
    
    q1 = FromAxisAngle({0,1,0}, 0);
    q2 = FromAxisAngle({0,1,0}, NKENTSEU_PI_DOUBLE);
    
    for(int frame = 0; frame < 60; frame++){
        double t = frame / 59.0;
        Quat q = Slerp(q1, q2, t);
        Mat4d R = FromRT(ToMat3(q), {0,0,0});
    
        img.Fill(255, 255, 255);
        std::vector<Vec3d> screen;
        for(auto v : cube){
            Vec4d p = P * (V * (R * v));
            screen.push_back(ProjectToScreen(p, width, height));
        }
    
        for(auto edge : edges)
            img.DrawLine((int)screen[edge.x].x, (int)screen[edge.x].y,
                         (int)screen[edge.y].x, (int)screen[edge.y].y, 255);
        img.SavePPM("Slerp_frame_TP11_"+std::to_string(frame)+".ppm");
    }
    
    for(int frame = 0; frame < 60; frame++){
        double t = frame / 59.0;
        Quat q = Lerp(q1, q2, t);
        Mat4d R = FromRT(ToMat3(q), {0,0,0});
    
        img.Fill(255, 255, 255);
        std::vector<Vec3d> screen;
        for(auto v : cube){
            Vec4d p = P * (V * (R * v));
            screen.push_back(ProjectToScreen(p, width, height));
        }
    
        for(auto edge : edges)
            img.DrawLine((int)screen[edge.x].x, (int)screen[edge.x].y,
                         (int)screen[edge.y].x, (int)screen[edge.y].y, 255);
        img.SavePPM("Lerp_frame_TP11_"+std::to_string(frame)+".ppm");
    }
}

// ============================================================
// TP12 : Décomposition SVD 3x3
// Validation reconstruction, orthogonalité et pseudo-inverse
// ============================================================
TEST_CASE(Loic_TP12, DecompositionSVD) {
    Mat3d A{}, Aplus{}, S{}, R{}, Iu{}, Iv{};
    SVD3x3 s;
    Vec3d b, x, r;

    // 20 matrices aléatoires : reconstruction et orthogonalité
    dist = std::uniform_real_distribution<double>(-1.0, 1.0);

    for(int i = 0; i < 20; i++){
        for(int j = 0; j < 3; j++)
            for(int k = 0; k < 3; k++)
                A(j, k) = dist(rng);

        s = svd3x3(A);

        S(0,0) = s.sigma.x;
        S(1,1) = s.sigma.y;
        S(2,2) = s.sigma.z;

        R = s.U * S * s.V.Transposed();

        // Reconstruction A = U * Sigma * Vt
        ASSERT_TRUE((R - A).norm() < 1e-6);

        // U et V orthogonaux
        Iu = s.U * s.U.Transposed();
        Iv = s.V * s.V.Transposed();
        ASSERT_TRUE((Iu - Mat3d::Identity()).norm() < 1e-6);
        ASSERT_TRUE((Iv - Mat3d::Identity()).norm() < 1e-6);

        // Valeurs singulières triées
        ASSERT_TRUE(s.sigma.x >= s.sigma.y);
        ASSERT_TRUE(s.sigma.y >= s.sigma.z);
    }

    // Matrice rang-déficiente : dernière σ ≈ 0
    A.setRow(0, {1,2,3});
    A.setRow(1, {2,4,6});
    A.setRow(2, {3,6,9});
    s = svd3x3(A);
    ASSERT_TRUE(std::fabs(s.sigma.z) < 1e-6);

    // Pseudo-inverse : moindres carrés
    A.setRow(0, {1, 2, 0});
    A.setRow(1, {3, 4, 0});
    A.setRow(2, {5, 6, 0});
    b = Vec3d(7, 8, 9);
    s = svd3x3(A);
    Aplus = s.pseudoInverse();
    x = Aplus * b;
    r = A * x - b;
    ASSERT_TRUE(r.Norm() < 1e-5);
}

// ============================================================
// TP13 : Résolution d'homographie (placeholder)
// Non implémenté — éléments du cours non encore couverts
// ============================================================
TEST_CASE(Loic_TP13, ResolutionHomographie) {
    (void)0;
}

// ============================================================
// TP14 : NkImage — dégradé, rectangle, diagonale, cercle
// ============================================================
TEST_CASE(Loic_TP14, NkImageFormes) {
    // Dégradé horizontal rouge / vertical vert
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            uint8_t r = (uint8_t)(255.0 * x / width);
            uint8_t g = (uint8_t)(255.0 * y / height);
            uint8_t b = 100;
            img.SetPixelRGBA(x, y, r, g, b);
        }
    }

    // Rectangle rouge
    for(int y = 120; y < 220; y++)
        for(int x = 120; x < 320; x++)
            img.SetPixelRGBA(x, y, 255, 0, 0);

    // Diagonale verte
    for(int i = 0; i < 640; i++)
        img.SetPixelRGBA(i, i, 0, 255, 0);

    // Cercle bleu
    int cx = 320, cy = 320, rad = 100;
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            int dx = x - cx, dy = y - cy;
            if(dx * dx + dy * dy < rad * rad)
                img.SetPixelRGBA(x, y, 0, 0, 255);
        }
    }

    img.SavePPM("loic_image_TP14.ppm");
}

// ============================================================
// TP15 : Convolution Gaussienne + Sobel (détection contours)
// ============================================================
TEST_CASE(Loic_TP15, ConvolutionGaussEtSobel) {
    std::vector<double> gaussian5 = {
        1, 4, 6, 4, 1,
        4,16,24,16, 4,
        6,24,36,24, 6,
        4,16,24,16, 4,
        1, 4, 6, 4, 1
    };
    for(auto& v : gaussian5) v /= 256.0;

    std::vector<double> sobelX = {
        -1,0,1,
        -2,0,2,
        -1,0,1
    };

    std::vector<double> sobelY = {
        -1,-2,-1,
        0, 0, 0,
        1, 2, 1
    };

    img.LoadPPM("input_arUCO.ppm");

    auto t0 = std::chrono::high_resolution_clock::now();

    NkImage blurred = img.Convolve(gaussian5, 5);
    NkImage gx = blurred.Convolve(sobelX, 3);
    NkImage gy = blurred.Convolve(sobelY, 3);
    NkImage edges_img = NkImage::CombineGradient(gx, gy);
    edges_img.SavePPM("loic_contours_TP15.ppm");

    auto t1 = std::chrono::high_resolution_clock::now();
    logger.Info("Convolution Gauss 5x5 + Sobel 3x3 : {0} ms",
        std::chrono::duration<double, std::milli>(t1-t0).count());
}

// ============================================================
// TP16 : Image intégrale et seuillage adaptatif (ArUCO)
// ============================================================
TEST_CASE(Loic_TP16, ImageIntegraleEtSeuillage) {
    NkImage img_local;
    img_local.LoadPPM("input_arUCO.ppm");

    // Seuillage adaptatif
    NkImage bin = AdaptiveThreshold(img_local, 31, 7);
    bin.SavePPM("loic_seuillage_TP16.ppm");

    // Mesure du temps de construction de l'image intégrale
    auto t0 = std::chrono::high_resolution_clock::now();
    IntegralImage ii(img_local.ToGrayscale(), img_local.Width(), img_local.Height());
    auto t1 = std::chrono::high_resolution_clock::now();

    logger.Info("Construction image intégrale : {0} ms",
        std::chrono::duration<double, std::milli>(t1-t0).count());
}
