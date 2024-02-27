// Gmsh project created on Tue Jan 10 15:22:37 2023
SetFactory("OpenCASCADE");
//+
Sphere(1) = {0, 1, 0, 1.0, -Pi/2, Pi/2, 2*Pi};
//+
//Sphere(2) = {-1, -3, 0, 1.0, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(2) = {3.8, 2.5, 0, 0.25 , -Pi/2, Pi/2, 2*Pi};
//+
//Sphere(4) = {3.8, -4.8, -0.1, 1.0, -Pi/2, Pi/2, 2*Pi};
//+
Physical Surface("A_2", 13) = {1};
//+
//Physical Surface("B", 14) = {2};
//+
//Physical Surface("C", 15) = {2};
//+
MeshSize {2, 4, 1, 3} = 0.05;
