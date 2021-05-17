#using XfromProjections
#using XfromProjections.curve_utils
#using XfromProjections.snake_forward
using LinearAlgebra
using IterTools

include("../estimate_angles.jl")
include("../utils.jl")
include("../../phantoms/sperm_phantom.jl")
include("../snake_forward.jl")
include("../snake.jl")
include("../curve_utils.jl")

using Plots
using Colors
using Random

Random.seed!(0)

function reconstruct(template, r, angles, bins, projection, max_iter, w, s, degree, k, length)
    #num_points = size(template, 1)
    centerline_points = recon2d_tail(deepcopy(template),r,angles,bins,projection,max_iter, s, w, degree)
    #residual = parallel_forward(get_outline(centerline_points, r)[1], angles, bins) - projection
    #centerline_points = keep_best_parts(residual, centerline_points, ang, bins, k, num_points, length, projection, r)
    #centerline_points = recon2d_tail(deepcopy(centerline_points),r,[ang],bins,projection,1000, s, w, degree)
    return centerline_points
end

function plot_update(curve, residual, name)
    #if norm(residual) < tolerance
    @info "plotted"
    label_txt = @sprintf "%s_%f" name norm(residual)
    plot!(curve[:,1], curve[:,2], aspect_ratio=:equal, label=label_txt, linewidth=2)
    #end
end

cwd = @__DIR__
savepath = normpath(joinpath(@__DIR__, "result"))
!isdir(savepath) && mkdir(savepath)
r(s) = 1.0
detmin, detmax = -36.5, 36.5
grid = collect(detmin:0.1:detmax)
images, tracks = get_sperm_phantom(301,r,grid)

bin_width = 0.125
bins = collect(detmin:bin_width:detmax)

ang = 0.0
angles, max_iter, stepsize = [ang], 10000, 0.1
tail_length = curve_lengths(tracks[end])[end]
num_points = 30

frames2reconstruct = collect(101:10:300)
reconstructions = zeros(num_points,2,length(frames2reconstruct)+1)
#Add actual track at the end so rand sperm works and we can compare timewise
centerline_points = tracks[frames2reconstruct[end]+10]
t = curve_lengths(centerline_points)
spl = ParametricSpline(t,centerline_points',k=1, s=0.0)
tspl = range(0, t[end], length=num_points)
reconstructions[:,:,end] = spl(tspl)'
tolerance = 2.0
k=8
s=0.0
degree=1
for (iter, frame_nr) in Base.Iterators.reverse(enumerate(frames2reconstruct))
    @info iter frame_nr

    #Get projection
    @info "making forward projection for frame: " frame_nr
    outline, normals = get_outline(tracks[frame_nr], r)
    projection = parallel_forward(outline, [ang], bins)

    head = tracks[frame_nr][1,:]
    #Add noise
    @info "adding gaussian noise at level 0.01"
    rho = 0.01
    e = randn(size(projection));
    e = rho*norm(projection)*e/norm(e);
    projection = projection + e;

    #Get ground truth of same length as recon
    t = curve_lengths(tracks[frame_nr])
    spl = ParametricSpline(t,tracks[frame_nr]', k=1, s=0.0)
    tspl = range(0, t[end], length=num_points)
    gt = spl(tspl)'

    angles = [ang]

    w_u = ones(num_points*2+2).*stepsize
    w_u[num_points+4:end] .= 0.0
    w_u[[1,2]] .=0.0
    #TEST
    w_u[num_points+1] = 0.0
    w_u[num_points+3] = 0.0

    w_l = ones(num_points*2+2).*stepsize
    w_l[1:num_points] .= 0.0
    w_l[end] = 0.0
    #TEST
    w_l[num_points+1] = 0.0
    w_l[num_points+3] = 0.0

    recon1_ok = false
    recon2_ok = false



    heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
    cd(savepath)
    @info "setting up template"
    rebinned_bins, rebinned_projection = rebin(projection[:,1], bins, 39)
    template = estimate_curve(rebinned_projection, rebinned_bins, angles, head, r, num_points)
    #template = get_straight_template(projection[:,1], r, [0.0 0.0], ang, num_points,bins)


    best_recon = deepcopy(template)
    best_cost = cost(template, ang, bins, r, projection[:,1])
    plot_update(template,best_cost, "template")

    @info "calculating initial reconstruction"
    #Reconstruct with weights only on one side

    #recon1 = recon2d_tail(deepcopy(template),r,[ang],bins,projection,1, 0.0, w_u, 1, doplot=true)



    for i = 1:10
        recon1 = reconstruct(best_recon, r, angles, bins, projection, max_iter, w_u, s, degree, k, tail_length)
        recon2 = reconstruct(best_recon, r, angles, bins, projection, max_iter, w_l, s, degree, k, tail_length)

        best_cost, best_recon[:,:], residual1, residual2 = try_improvement(best_cost, recon1, recon2, ang, bins, projection, best_recon, tail_length, r)

        label_txt1 = recon1
        label_txt2 = recon2
        plot_update(recon2, residual1, label_txt1)
        plot_update(recon1, residual2, label_txt2)
        plot_update(best_recon, best_cost, "initial")


        @info "checking if any parts could need mirroring"
        initial1 = deepcopy(best_recon)
        initial2 = deepcopy(best_recon)

        for flip_pt=1:num_points
            recon1_flipped = flip(initial1,flip_pt,ang)
            recon2_flipped = flip(initial2,flip_pt,ang)

            #best_cost, best_recon[:,:], residual1, residual2 = try_improvement(best_cost, recon1_flipped, recon2_flipped, ang, bins, projection, best_recon, tail_length, r)

            #mirror and reconstruct with weights on both sides
            recon1 = reconstruct(recon1_flipped, r, angles, bins, projection, 100, w_u, s, degree, k, tail_length)
            recon2 = reconstruct(recon1_flipped, r, angles, bins, projection, 100, w_l, s, degree, k, tail_length)
            best_cost, best_recon[:,:], residual1, residual2 = try_improvement(best_cost, recon1, recon2, ang, bins, projection, best_recon, tail_length, r)

            recon1 = reconstruct(recon2_flipped, r, angles, bins, projection, 100, w_u, s, degree, k, tail_length)
            recon2 = reconstruct(recon2_flipped, r, angles, bins, projection, 100, w_l, s, degree, k, tail_length)
            best_cost, best_recon[:,:], residual1, residual2 = try_improvement(best_cost, recon1, recon2, ang, bins, projection, best_recon, tail_length, r)
        end
        plot_update(best_recon, best_cost, "flip1")
        # if best_cost < 1.0
        #     break
        # end

        # for (flip_i,flip_j) in subsets(1:num_points, Val{2}())
        #     recon1_flipped = flip(initial1,flip_i,ang)
        #     recon1_flipped = flip(recon1_flipped,flip_j,ang)
        #     recon2_flipped = flip(initial2,flip_i,ang)
        #     recon2_flipped = flip(recon2_flipped,flip_j,ang)
        #
        #     best_cost, best_recon[:,:], residual1, residual2 = try_improvement(best_cost, recon1_flipped, recon2_flipped, ang, bins, projection, best_recon, tail_length, r)
        #
        #     recon1 = reconstruct(recon1_flipped, r, angles, bins, projection, 100, w_u, s, degree, k, tail_length)
        #     recon2 = reconstruct(recon1_flipped, r, angles, bins, projection, 100, w_l, s, degree, k, tail_length)
        #     best_cost, best_recon[:,:], residual1, residual2 = try_improvement(best_cost, recon1, recon2, ang, bins, projection, best_recon, tail_length, r)
        #
        #     recon1 = reconstruct(recon2_flipped, r, angles, bins, projection, 100, w_u, s, degree, k, tail_length)
        #     recon2 = reconstruct(recon2_flipped, r, angles, bins, projection, 100, w_l, s, degree, k, tail_length)
        #     best_cost, best_recon[:,:], residual1, residual2 = try_improvement(best_cost, recon1, recon2, ang, bins, projection, best_recon, tail_length, r)
        # end
        # plot_update(best_recon, best_cost, "flip2")
    end
    reconstructions[:,:,iter] = best_recon
    #plot!(best_recon[:,1], best_recon[:,2], aspect_ratio=:equal, label=best_cost, linewidth=2)
    savefig(@sprintf "heuristic_result_%d" frame_nr)
    heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
    plot!(best_recon[:,1], best_recon[:,2], aspect_ratio=:equal, linewidth=2)
    mirror = flip(best_recon,1,ang)
    plot!(mirror[:,1], mirror[:,2], aspect_ratio=:equal, linewidth=2)
    savefig(@sprintf "result_%d" frame_nr)
end

global ps = AbstractPlot[]
for (iter, frame_nr) in enumerate(frames2reconstruct)
    best_recon = reconstructions[:,:,iter]
    mirror = flip(best_recon,1,ang)

    l = @layout [a b c]
    p = heatmap(grid, grid, Gray.(images[:,:,frame_nr]), yflip=false, aspect_ratio=:equal, framestyle=:none, legend=false)
    plot!(best_recon[:,1],best_recon[:,2], aspect_ratio=:equal, linewidth=5)
    plot!(mirror[:,1], mirror[:,2], aspect_ratio=:equal, linewidth=5)
    push!(ps,p)
    if length(ps) == 3
        plot(ps[1], ps[2], ps[3], layout = l, size=(2000,600), linewidth=5)
        savefig(@sprintf "result_all_%d" frame_nr)
        global ps = AbstractPlot[]
    end
end
cd(cwd)
