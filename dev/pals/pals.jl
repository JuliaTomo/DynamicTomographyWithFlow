using ParamLevelSet
using jInv.Mesh
using SparseArrays
using LinearAlgebra

function init_pals(vol_size, nbasis=4)
	half = vol_size ./ 2
    # Mesh = getRegularMesh([-half[1];half[1];-half[2];half[2];-half[3];half[3]],[vol_size[1],vol_size[2],vol_size[3]]);
    Mesh = getRegularMesh([1;vol_size[1];1;vol_size[2];1;vol_size[3]],[vol_size[1],vol_size[2],vol_size[3]]);
	
	alpha = [1.5;2.5;-2.0;-1.0];
	beta = [2.5;2.0;-1.5;2.5];
	Xs = [0.5 0.5 0.5; 2.0 2.0 2.0; 1.2 2.3 1.5; 2.2 1.5 2.0] .* vol_size[1] .* 0.3
	m = wrapTheta(alpha,beta,Xs);	
	
	return Mesh, m
end

function recon2d_slices_pals!(u::Array{T, 3}, A, b, niter::Int, nbasis::Int=4) where {T <: AbstractFloat}
    
    mesh, m = init_pals(size(u), nbasis)
    sigmaH = getDefaultHeaviside();
    phat = similar(p)
    r = similar(p)
    At = A'
    stepsize = 1.0
    nslice = size(p,2)

    HW = size(u,1)*size(u,2)
    szpslice = size(p,1)*size(p,3)
    J = spzeros(prod(size(p)), nbasis*5) # A * J0

    for i=1:niter
        u0, JBuilder = MeshFreeParamLevelSetModelFunc(mesh,m;computeJacobian=1, sigma=sigmaH)
        J0 = getSparseMatrix(JBuilder)
    
        # construct J
        Threads.@threads for slice=1:nslice
            J[szpslice*(slice-1)+1:szpslice*slice, :] .= A * J0[HW*(slice-1)+1:HW*slice, :]
    
            u_slice = view(u, :, :, slice)
            p_vec = view(phat, :,slice,:)
            @views p_vec_ = vec(p_vec)
    
            @views mul!(p_vec_, A, vec(u_slice))
            @views r[:,slice,:] .= p_vec .- p[:,slice,:]
        end
    
        Jt = J'
        Hessian = Jt*J
        Hinv = pinv(Hessian)
        grad = Jt * vec(r)
        m .-= stepsize * Hinv * grad

        fres = sum(r*r) / length(r)
        println("residual: $fres")
    end
        
    return _recon2d_slices_tvrdart!(u, A, b, niter, w_tv, param)
end