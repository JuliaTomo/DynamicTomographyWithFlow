include("./utils.jl")
include("../../phantoms/sperm_phantom.jl")
include("../snake_forward.jl")
include("../snake.jl")
include("../curve_utils.jl")
using IterTools
using LinearAlgebra

function count_parallel(projection, r; limit=30)
    #get the 'end points' of the projection, by getting the first and last value where value is greater than tail diameter, which is minimum
    projection_end1 = findfirst(p -> p > 2*r(0.0), projection)
    projection_end2 = findlast(p -> p > 2*r(0.0), projection)

    first = min(projection_end1, projection_end2)
    last = max(projection_end1, projection_end2)
    all_parallel = findall(p -> p <= 2.0*r(0.0), projection[first:last])
    neighbor_diff = all_parallel[2:end]-all_parallel[1:end-1]
    changes = count(c -> c > limit, neighbor_diff)+1
    return changes, all_parallel
end

images, tracks = get_sperm_phantom(301,grid_size=0.1)

r(s) = 1.0
detmin, detmax = -38.0, 38.0
bins = collect(detmin:0.125:detmax)

cwd = @__DIR__
savepath = normpath(joinpath(@__DIR__, "result"))
!isdir(savepath) && mkdir(savepath)

cd(savepath)
c = []
b = []
for (i,track) in enumerate(tracks)
    outline, normals = get_outline(track, r)
    projection = parallel_forward(outline, [π/2], bins)

    changes, a = count_parallel(projection[:,1], r, limit=30)
    curvature_changes, k, length_ok, changes, prime = could_be_sperm_tail(30.0, track, π/2, smoothness=0.1)
    plot(track[:,1], track[:,2], aspect_ratio=:equal, label=(@sprintf "%d_%f" changes maximum(abs.(prime))))
    push!(c, curvature_changes)
    push!(b, norm(prime))
    savefig(@sprintf "result_all_%d" i)
end
