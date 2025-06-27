lc = 1.5;

Point(1) = {0.0, 0.0, 0.01, lc};
Point(2) = {0.01,   0.0, 0.01, lc};
Point(3) = {0.01,   0.0, 0.0,   lc};
Point(4) = {0.0,   0.0, 0.0,   lc};

Line(1) = {4, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};

Curve Loop(1) = {2, 3, 4, 1};
Plane Surface(1) = {1};
//+
Extrude {0, 0.05, 0} {
  Surface{1};
}

//Physical Surface("sides", 27) = {21, 17, 13, 25};
Physical Surface("top", 28) = {21, 17, 13, 25, 26, 1};
//Physical Surface("bottom", 29) = {1};

Physical Volume("body", 30) = {1};

Transfinite Curve {8, 4, 7, 3, 6, 9, 2, 1} = 20 Using Progression 1;
Transfinite Curve {11, 12, 16, 20} = 100 Using Progression 1;

Transfinite Surface {21};
Transfinite Surface {13};
Transfinite Surface {17};
Transfinite Surface {25};
Transfinite Surface {26};
Transfinite Surface {1};

Recombine Surface {21, 17, 26, 1, 13, 25};

Transfinite Volume{1};
