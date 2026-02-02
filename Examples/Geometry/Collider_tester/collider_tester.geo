// Gmsh project
SetFactory("OpenCASCADE");

sph_radius = 0.5;
displacement = 0.05;
rod_radius = 0.15;
//+
Box(1) = {0, 0, 0, 1, 1, 1};
//+
Sphere(2) = {1+sph_radius+displacement, 0.5, 0.5, sph_radius, -Pi/2, Pi/2, 2*Pi};
//+
Cylinder(3) = {0.6, 0.5, 0.5, 1, 0, 0, rod_radius, 2*Pi};
//+
BooleanDifference{ Volume{3}; Delete;  }{ Volume{2}; }

BooleanDifference{ Volume{3}; Delete;  }{ Volume{1}; }

Coherence;

//+
Physical Surface("Cube_A", 34) = {14, 11, 15, 16, 12, 9, 13};
//+
Physical Surface("Cube_B", 35) = {17, 10};
//+
Physical Surface("Joint", 36) = {8, 9, 10};
