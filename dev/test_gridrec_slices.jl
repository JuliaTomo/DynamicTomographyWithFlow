using DSP
using FFTW
# make a synthetic projection
using TomoForward
img = zeros(128, 128, 2)
H, W, nslice = size(img)
img[10:70, 40:60, 1:2] .= 1.0


nangles = 90
detcount = 128
# detcount = Int(floor(size(img,1)*1.4))
angles = Array(LinRange(0,pi,nangles+1)[1:nangles])
proj_geom = ProjGeom(1.0, detcount, angles)

isdefined_A = @isdefined A
if isdefined_A == false
    A = fp_op_parallel2d_strip(proj_geom, size(img, 1), size(img, 2))
end

p = zeros(nangles, nslice, detcount)
for i=1:nslice
    p[:,i,:] = reshape(A * vec(img[:,:,i]), nangles, detcount)
end

pp = p[:, 1, :]
pp_fft = fftshift( fft( ifftshift(pp, 2) ), 2 )
# q = recon2d_slices_gridrec(p, angles)

srcx = W/2 - 0.5 + cos()


using PyPlot
# using Plots
# plot(Gray.(q[:,:,:]))