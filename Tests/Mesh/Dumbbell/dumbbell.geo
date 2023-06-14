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
MeshSize {2, 3, 6, 1, 4, 7, 5, 8} = 0.025;
