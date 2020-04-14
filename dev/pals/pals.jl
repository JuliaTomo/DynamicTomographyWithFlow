using ParamLevelSet
using jInv.Mesh
include("pals.jl")

function init_pals(vol_size, nbasis=4)
	half = vol_size ./ 2
	Mesh = getRegularMesh([-half[1];half[1];-half[2];half[2];-half[3];half[3]],[vol_size[1],vol_size[2],vol_size[3]]);
	
	alpha = [1.5;2.5;-2.0;-1.0];
	beta = [2.5;2.0;-1.5;2.5];
	Xs = [0.5 0.5 0.5 ; 2.0 2.0 2.0; 1.2 2.3 1.5; 2.2 1.5 2.0];
	m = wrapTheta(alpha,beta,Xs);	
	
	return Mesh, m
end

