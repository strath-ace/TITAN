// Gmsh project created on Wed Jun 28 15:18:06 2023
SetFactory("OpenCASCADE");
//+
Cylinder(1) = {0.1, 0, 0, 0.3, 0, 0, 0.02, 2*Pi};
//+
Cylinder(2) = {0.05, 0, 0, 0.4, 0, 0, 0.025, 2*Pi};


//+
BooleanDifference{ Volume{2}; Delete; }{ Volume{1}; Delete; }
//+
MeshSize {4, 2, 1, 3} = 0.01;
//+
Physical Surface("Rod_in", 7) = {3, 1, 2};
//+
Physical Surface("Rod_out", 8) = {6, 4, 5};
