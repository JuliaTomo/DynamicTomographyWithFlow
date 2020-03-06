using TomoForward
using Images
using Plots
using XfromProjections
using StaticArrays
using PyCall
using Logging

include("./simple_phantoms.jl")
include("./tv_primaldual_flow.jl")
include("./optical_flow.jl")
include("./sperm_phantom.jl")

H,W = 153, 153
function radon_operator(img, detcount)
    nangles = 4
    angles = range(0.0,π, length=nangles)#rand(0.0:0.001:π, nangles)#
    proj_geom = ProjGeom(0.5, detcount, angles)
    A = fp_op_parallel2d_line(proj_geom, size(img,1), size(img,2), -38.0,38.0, -38.0,38.0)
    return A
end

frames = get_sperm_phantom(10,grid_size=0.5)
detcount = Int(floor(size(frames[:,:,1],1)*1.4))
#match size of input image (generating data)
As = map(t -> radon_operator(frames[:,:,t], detcount),1:size(frames)[3])
bs = zeros(size(As[1])[1],size(frames)[3])
map(t -> bs[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])
niter=100
u0s = zeros(H,W,size(frames)[3])

As  = map(t -> radon_operator(u0s[:,:,t], detcount),1:size(u0s)[3])

w_tv = 0.3
w_flow  = 0.1

@info "Reconstructing using joint motion estimation and reconstruction"
c=10.0
us_flow = recon2d_tv_primaldual_flow(As, bs, u0s, 20, niter, w_tv, w_flow, c)

@info "Reconstruction using tv regularization frame by frame"
us_tv = zeros(H,W,size(frames)[3])
for t = 1:size(frames)[3]
    A = As[t]
    p = bs[:,t]
    u0 = u0s[:,:,t]
    us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
end

@info "Preparing results in human readable format"
anim = @animate for t=1:size(frames)[3]
    l = @layout [a b c]
    p1 = plot(Gray.(frames[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="Ground truth", yflip=false)
    p2 = plot(Gray.(us_flow[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="Flow", yflip=false)
    p3 = plot(Gray.(us_tv[:,:,t]), aspect_ratio=:equal, framestyle=:none, title="TV", yflip=false)
    plot(p1, p2, p3, layout = l)
end

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)
gif(anim, "reconstruction_flow.gif", fps = 1)
cd(cwd)
