PrintNumber[z_, n_Integer?Positive, label_String : ""] := Print[label, ToString[NumberForm[N[z, n], n]]];

SphericalIn[n_, z_] := BesselI[n + 1/2, z] * Sqrt[Pi / (2*z)]
SphericalKn[n_, z_] := BesselK[n + 1/2, z] * Sqrt[Pi / (2*z)]

digits = 24
PrintNumber[SphericalIn[0, 1], digits, "SphericalIn[0, 1]         = "]
PrintNumber[SphericalIn[6, -3/2 + 2I], digits, "SphericalIn[6, -3/2 + 2I] = "]
PrintNumber[SphericalKn[0, 1], digits, "SphericalKn[0, 1]         = "]
PrintNumber[SphericalKn[6, -3/2 + 2I], digits, "SphericalKn[6, -3/2 + 2I] = "]
