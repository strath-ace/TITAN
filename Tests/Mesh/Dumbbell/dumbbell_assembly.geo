// Gmsh project created on Mon May  8 14:31:51 2023
SetFactory("OpenCASCADE");
//+
Point(1) = {-0.5, 0, 0, 1.0};
//+
Point(2) = {-0.5, 0.5, 0, 1.0};
//+
Point(3) = {-0.4, 0.5, 0, 1.0};
//+
Point(4) = {-0.4, 0.05, 0, 1.0};
//+
Point(5) = {0.4, 0.05, 0, 1.0};
//+
Point(6) = {0.4, 0.5, 0, 1.0};
//+
Point(7) = {0.5, 0.5, 0, 1.0};
//+
Point(8) = {0.5, 0.0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Extrude {{1, 0, 0}, {0, 0, 0}, 2*Pi} {
  Curve{1}; Curve{2}; Curve{3}; Curve{4}; Curve{5}; Curve{6}; Curve{7}; 
}


//+
Curve Loop(10) = {10};
//+
Plane Surface(8) = {10};
//+
Curve Loop(11) = {11};
//+
Plane Surface(9) = {11};
//+

//+
Coherence;
//+
MeshSize {1, 4, 2, 3, 5, 8, 6, 7} = 0.05;
//+
Physical Surface("bell1", 14) = {3, 2, 1, 8};
//+
Physical Surface("bell2", 15) = {7, 5, 6, 9};
//+
Physical Surface("rod", 16) = {4, 9, 8};
