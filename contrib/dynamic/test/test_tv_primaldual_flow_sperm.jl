using TomoForward
using Images
using Plots
using XfromProjections
using StaticArrays
using PyCall
using Logging

include("../tv_primaldual_flow.jl")
include("../optical_flow.jl")
include("../../phantoms/sperm_phantom.jl")

H,W = 609, 609
detmin, detmax = -38.0, 38.0
function radon_operator(height, width, detcount, angles)
    proj_geom = ProjGeom(0.5, detcount, angles)
    A = fp_op_parallel2d_line(proj_geom, height, width, detmin,detmax, detmin,detmax)
    return A
end
grid = collect(detmin:0.1:detmax)
r(s) = 1.0

images, tracks = get_sperm_phantom(301,r,grid)
frames = images[:,:,collect(271:10:300)]

w_tv = 0.01
w_flow  = 0.001
c=10.0


detcount = Int(floor(H*1.4))

cwd = @__DIR__
path = normpath(joinpath(@__DIR__, "result"))
!isdir(path) && mkdir(path)
cd(path)

angle_numbers = [2,4]
for ang_nr in angle_numbers
    angles = collect(range(π/2,3*π/2,length=ang_nr+1))[1:end-1]

    #match size of input image (generating data)
    As = map(t -> radon_operator(size(frames[:,:,1],1),size(frames[:,:,1],2),detcount, angles),1:size(frames)[3])
    bs = zeros(size(As[1])[1],size(frames)[3])
    map(t -> bs[:,t] = As[t]*vec(frames[:,:,t]), 1:size(frames)[3])

    u0s = zeros(H,W,size(frames)[3])

    As  = map(t -> radon_operator(H,W, detcount, angles),1:size(u0s)[3])

    @info "Reconstruction using tv regularization frame by frame"
    niter=450
    us_tv = zeros(H,W,size(frames)[3])
    for t = 1:size(frames)[3]
        A = As[t]
        p = bs[:,t]
        u0 = u0s[:,:,t]
        us_tv[:,:,t] .= recon2d_tv_primaldual!(us_tv[:,:,t], A, p, niter, w_tv, c)
    end

    @info "Reconstructing using joint motion estimation and reconstruction"
    niter1=50
    niter2=20
    u0s = deepcopy(us_tv)
    us_flow = recon2d_tv_primaldual_flow(As, bs, u0s, niter1, niter2, w_tv, w_flow, c)

    p1 = heatmap(grid, grid, Gray.(us_flow[:,:,1]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
    savefig(@sprintf "test_%d" length(angles))
    plot!(tracks[271][:,1], tracks[271][:,2], aspect_ratio=:equal, linewidth=5)
    p2 = heatmap(grid, grid, Gray.(us_flow[:,:,2]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
    plot!(tracks[281][:,1], tracks[281][:,2], aspect_ratio=:equal, linewidth=5)
    p3 = heatmap(grid, grid, Gray.(us_flow[:,:,3]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
    plot!(tracks[291][:,1], tracks[291][:,2], aspect_ratio=:equal, linewidth=5)
    l = @layout [a b c]
    plot(p1, p2, p3, layout = l, size=(2000,600), linewidth=5)
    savefig(@sprintf "result_all_%d" length(angles))
end

cd(cwd)
