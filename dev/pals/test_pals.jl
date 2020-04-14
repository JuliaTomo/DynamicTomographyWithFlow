using TomoForward
using XfromProjections

# using MKLSparse # uncomment if you've installed MKLSparse, which will boost the performance

# test slice by slice
img = zeros(100, 100, 128)
img[40:70, 40:60, 50:70] .= 1.0

nslice = size(img, 3)

nangles = 90
detcount = Int(floor(size(img,1)*1.4))
proj_geom = ProjGeom(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles])

isdefined_A = @isdefined A
if isdefined_A == false
    A = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
end

p = zeros(nangles, nslice, detcount)
for i=1:nslice
    p[:,i,:] = reshape(A * vec(img[:,:,i]), nangles, detcount)
end

u = zeros(size(img))

mesh, m = init_pals(size(img))
sigmaH = getDefaultHeaviside();
phat = similar(p)
r = similar(p)
At = A'
atr = similar(img)
stepsize = 0.1
nbasis = 4

using LinearAlgebra
using SparseArrays
grad = zeros(nbasis*5)
H = zeros(nbasis*5, nbasis*5)

HW = size(img,1)*size(img,2)
szpslice = size(p,1)*size(p,3)
J = spzeros(prod(size(p)), nbasis*5) # A * J0

for i=1:1
	u0, JBuilder = MeshFreeParamLevelSetModelFunc(mesh,m;computeJacobian=1, sigma=sigmaH)
	J0 = getSparseMatrix(JBuilder)

	# construct J
	for slice=1:nslice
		J[szpslice*(slice-1)+1:szpslice*slice, :] .= A * J0[HW*(slice-1)+1:HW*slice, :]

		u_slice = view(u, :, :, slice)
		p_vec = view(phat, :,slice,:)

		@views mul!(vec(p_vec), A, vec(u_slice))
		@views r[:,slice,:] .= p_vec - p[:,slice,:]
	end

	Jt = J'
	Hessian = Jt*J
	Hinv = pinv(Hessian)
	grad = Jt * vec(r)
	m .-= stepsize * Hinv * grad
end


# u0,JBuilder = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=1,sigma=sigmaH,bf = bf);
# J0 = getSparseMatrix(JBuilder);

# Kernel(r) = max.(1.0-r, 0) .^8 .* (32*r.^3 .+ 25*r.^2 .+ 8*r .+ 1)

# switch rtype
#     case 'global'          % Global RBF (Gaussian)
#         kernelM = @(r) exp(-r.^2);
#         ki = 3.3;
#     case 'compact'          % Compactly-supported RBF (Wendland C4)
#         kernelM = @(r) max(1-r,0).^8.*(32*r.^3 + 25*r.^2 + 8*r + 1);
#         ki = 1;