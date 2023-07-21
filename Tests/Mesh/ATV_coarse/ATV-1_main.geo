//+
SetFactory("OpenCASCADE");
//+
Cone(1) = {0.0, 0.0, 0.0, -0.745, 0, 0, 0.625, 1.114, 2*Pi};
//+
Cone(2) = {-0.745, 0.0, 0.0, -0.458, 0, 0, 1.114, 2.25, 2*Pi};
//+
Cylinder(3) = {-1.203, 0.0, 0.0, -4.784, 0, 0, 2.25, 2*Pi};
//+
Cone(4) = {-5.987, 0.0, 0.0, -1.441, 0, 0, 2.25, 2.03, 2*Pi};
//+
Cylinder(5) = {-7.428, 0.0, 0.0, -1.957, 0, 0, 2.03, 2*Pi};
//+
Cone(6) = {-9.385, 0.0, 0.0, -0.359, 0, 0, 2.03, 1.62, 2*Pi};
//+

// Joint specifics
// Center coordinate pulled back a bit so that the joint goes into the main body!!
// For easier dealing of intersections!
// Adjust the y-coordinate of the cylinder joint center and increase length by same amount!
// perfect center on the main body = (-7.712, 2.03, 0.0) & perfect joint length = 2.103
Cylinder(7) = {-7.712, 1.53, 0.0, 0.0, 2.903, 0, 0.027, 2*Pi};
//+

Rotate {{0, 1, 0}, {-7.712, 1.53, 0.0}, Pi} {
  Duplicata { Volume{7};}
}

//Characteristic Length {14, 13} = 0.01;
//+
// Rotations of the above created joint!
//+
Rotate {{1, 0, 0}, {0, 0, 0}, 11*Pi/90} {
  Duplicata { Volume{7}; Volume{8} ;}
}
//+
Rotate {{1, 0, 0}, {0, 0, 0}, -11*Pi/90} {
  Duplicata { Volume{7}; Volume{8} ;}
}
//+
Rotate {{1, 0, 0}, {0, 0, 0}, 79*Pi/90} {
  Duplicata { Volume{7}; Volume{8} ;}
}
//+
Rotate {{1, 0, 0}, {0, 0, 0}, -79*Pi/90} {
  Volume{7}; Volume{8} ;
}
//+


// Creating the solar panels similar to that of the cylindrical joint
// Solar panels facing the flow for increased drag during re-entry!
//+
Box(15) = {-7.752, 4.133, -0.579, 0.080, 7.025, 1.158};

// Rotating the volumes of the solar panels
//+
Rotate {{1, 0, 0}, {0, 0, 0}, 11*Pi/90} {
  Duplicata {Volume{15};}
}
//+
Rotate {{1, 0, 0}, {0, 0, 0}, -11*Pi/90} {
  Duplicata {Volume{15};}
}
//+
Rotate {{1, 0, 0}, {0, 0, 0}, 79*Pi/90} {
  Duplicata {Volume{15};}
}
//+
Rotate {{1, 0, 0}, {0, 0, 0}, -79*Pi/90} {
  Volume{15};
}
//+
Coherence;
//+
Characteristic Length {6, 15, 3, 11, 4, 17, 5, 13} = 0.005;
//+
// Physical Surface("Panel1") = {45, 25, 46, 48, 49, 44, 47};
// Physical Surface("Panel2") = {21, 38, 39, 40, 41, 42, 43};
// Physical Surface("Panel3") = {17, 32, 33, 34, 35, 36, 37};
// Physical Surface("Panel4") = {29, 50, 51, 52, 53, 54, 55};
// Physical Surface("ATVmain") = {12, 13, 14, 15, 1, 58, 56, 59, 61, 63, 64, 65};
// Physical Surface("Joint1") = {15, 27, 25};
// Physical Surface("Joint2") = {14, 23, 21};
// Physical Surface("Joint3") = {12, 19, 17};
//Physical Surface("Joint4") = {13, 31, 29};
//+
BooleanUnion{ Volume{21}; Delete; }{ Volume{14}; Delete; }
//+
BooleanUnion{ Volume{22}; Delete; }{ Volume{16}; Delete; }
//+
BooleanUnion{ Volume{23}; Delete; }{ Volume{18}; Delete; }
//+
BooleanUnion{ Volume{20}; Delete; }{ Volume{12}; Delete; }
//+
BooleanUnion{ Volume{6}; Delete; }{ Volume{7}; Volume{4}; Delete; }
//+
BooleanUnion{ Volume{28}; Delete; }{ Volume{3}; Volume{2}; Volume{1}; Delete; }
//+
BooleanUnion{ Volume{28}; Delete; }{ Volume{11}; Volume{10}; Volume{9}; Volume{8}; Delete; }
//+


//+
BooleanUnion{ Surface{104}; Delete; }{ Surface{105}; Surface{106}; Surface{107}; Surface{108}; Surface{109}; Delete; }
//+
BooleanUnion{ Surface{105}; Delete; }{ Surface{106}; Delete; }
//+
BooleanUnion{ Surface{107}; Delete; }{ Surface{106}; Delete; }
//+

//+
MeshSize {110, 109, 108, 107, 95, 94, 91, 90, 106, 103, 102, 99, 92, 98, 93, 100, 33, 31, 101, 21, 19, 89, 88, 5, 6, 97, 3, 4, 96, 77, 8, 7, 76, 10, 9, 85, 84, 23, 25, 72, 27, 29, 73, 80, 78, 81, 79, 74, 75, 86, 87, 82, 83, 105, 104} = 0.25;

Coherence;
//+

//+
Field[1] = Ball;
Field[1].Radius = 0.3;
Field[1].VIn = 0.02;
Field[1].VOut = 1000;
Field[1].XCenter = -7.712;
Field[1].YCenter = -1.882;
Field[1].ZCenter = -0.76;

Field[2] = Ball;
Field[2].Radius = 0.3;
Field[2].VIn = 0.02;
Field[2].VOut = 1000;
Field[2].XCenter = -7.712;
Field[2].YCenter = 1.882;
Field[2].ZCenter = -0.76;

Field[3] = Ball;
Field[3].Radius = 0.3;
Field[3].VIn = 0.02;
Field[3].VOut = 1000;
Field[3].XCenter = -7.712;
Field[3].YCenter = -1.882;
Field[3].ZCenter = 0.76;

Field[4] = Ball;
Field[4].Radius = 0.3;
Field[4].VIn = 0.02;
Field[4].VOut = 1000;
Field[4].XCenter = -7.712;
Field[4].YCenter = 1.882;
Field[4].ZCenter = 0.76;

Field[5] = Ball;
Field[5].Radius = 0.2;
Field[5].VIn = 0.015;
Field[5].VOut = 1000;
Field[5].XCenter = -7.712;
Field[5].YCenter = 3.831;
Field[5].ZCenter = 1.553;

Field[6] = Ball;
Field[6].Radius = 0.2;
Field[6].VIn = 0.015;
Field[6].VOut = 1000;
Field[6].XCenter = -7.712;
Field[6].YCenter = 3.831;
Field[6].ZCenter = -1.553;

Field[7] = Ball;
Field[7].Radius = 0.2;
Field[7].VIn = 0.015;
Field[7].VOut = 1000;
Field[7].XCenter = -7.712;
Field[7].YCenter = -3.831;
Field[7].ZCenter = 1.553;

Field[8] = Ball;
Field[8].Radius = 0.2;
Field[8].VIn = 0.015;
Field[8].VOut = 1000;
Field[8].XCenter = -7.712;
Field[8].YCenter = -3.831;
Field[8].ZCenter = -1.553;


Cylinder(200) = {-7.712, 2.04, 0.0, 0.0, 2.083, 0, 0.0216, 2*Pi};

Rotate {{0, 1, 0}, {-7.712, 1.53, 0.0}, Pi} {
  Duplicata { Volume{200};}
}

Rotate {{1,0, 0}, {0, 0, 0.0}, 11*Pi/90} {
  Duplicata { Volume{200};Volume{201};}
}

Rotate {{1,0, 0}, {0, 0, 0.0}, -11*Pi/90} {
  Duplicata { Volume{200};Volume{201};}
}


Rotate {{1,0, 0}, {0, 0, 0.0}, 79*Pi/90} {
  Duplicata { Volume{200};Volume{201};}
}

Rotate {{1,0, 0}, {0, 0, 0.0}, -79*Pi/90} {
 Volume{200};Volume{201};
}



Coherence;

Transfinite Curve {129,130,131,133,137,139,141,138,145,147,146,149,148,152,150,151,126,127,124,128} = 150 Using Bump 0.2;
//+
Transfinite Curve {33, 29, 5, 4, 57, 53, 7, 6, 9, 8, 41, 37, 49, 45, 11, 10, 136, 132,134,135,142,143,140,144,121,122,123,125} = 25 Using Progression 1.00;

Transfinite Surface {82};
Transfinite Surface {83};
Transfinite Surface {84};
Transfinite Surface {85};
//+
Transfinite Surface {88};
Transfinite Surface {89};
Transfinite Surface {90};
Transfinite Surface {91};

Transfinite Surface {94};
Transfinite Surface {95};
Transfinite Surface {96};
Transfinite Surface {97};

Transfinite Surface {76};
Transfinite Surface {77};
Transfinite Surface {78};
Transfinite Surface {79};
//+

//+
Field[9] = Min;
//+
Field[9].FieldsList = {1, 2, 3, 4,5,6,7,8};
//+
Background Field = 9;

//+
Physical Surface("Panel1", 211) = {48, 44, 46, 47, 49, 45, 28};
//+
Physical Surface("Panel2", 212) = {61, 58, 60, 59, 56, 57, 40};
//+
Physical Surface("Panel3", 213) = {64, 67, 62, 65, 66, 63, 22};
//+
Physical Surface("Panel4", 214) = {54, 55, 50, 52, 53, 51, 34};
//+
Physical Surface("Joint1", 215) = {28, 18, 82, 83};
//+
Physical Surface("Joint1_inner", 216) = {85, 84, 87, 86};
//+
Physical Surface("Joint2_inner", 217) = {99, 97, 96, 98};
//+
Physical Surface("Joint3_inner", 218) = {80, 78, 79, 81};
//+
Physical Surface("Joint4_inner", 219) = {93, 91, 90, 92};
//+
Physical Surface("Joint4", 220) = {19, 89, 88, 34};
//+
Physical Surface("Joint3", 221) = {16, 77, 76, 22};
//+
Physical Surface("Joint2", 222) = {94, 95, 40, 17};
//+
Physical Surface("ATV", 223) = {73,75, 74, 72, 71, 70, 68, 69, 16, 17, 19, 18};
