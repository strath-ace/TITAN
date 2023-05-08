// Gmsh project created on Mon Oct  3 17:21:19 2022
SetFactory("OpenCASCADE");
//+
Sphere(1) = {0, 0, 0, 1.0, -Pi/2, Pi/2, 2*Pi};
//+
MeshSize {1, 2} = 0.1;
