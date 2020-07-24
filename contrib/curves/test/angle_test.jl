using Plots
include("../estimate_angles.jl")
include("../snake_forward.jl")
include("../../phantoms/sperm_phantom.jl")
using ResumableFunctions
using LsqFit
using Statistics


######################PREPROCESS#############################
r(s) = 1.0
global images, CL = get_sperm_phantom(400,r,collect(-36.5:0.1:36.5)))

function process_frame(frame)
    #translate so it starts at zero
    center_line = CL[frame].-CL[frame][1,:]'

    #find angles and rotatet the tail so it is on average parallel with the x-axis
    # this is done by converting points to complex numbers, finding the angles, taking average and turning -average
    head = center_line[1,:]
    cks = get_ck(center_line)

    xy = get_xy(cks)[2:end,:]

    angles_est = angle.(cks[2:end])
    average_angle = sum(angles_est)/length(angles_est)
    center_line = rotate_points(xy, -average_angle)
    plot(center_line[:,1], center_line[:,2], label="center line", aspect_ratio=:equal, legend=true)
    ################################################################################################


    #Define thickness function - same as in phantom TODO pass r to phantom function
    r(t) = 1.0

    #Determine the outline
    outline = get_outline(center_line, r)[1]
    plot!(outline[:,1], outline[:,2], label = "outline")

    #Define angles and bin  width for forward projection
    detmin, detmax = -36.5, 36.5
    bin_width = 0.125
    bins = collect(detmin:bin_width:detmax)
    angles = [0.0]
    projection = parallel_forward(outline,angles,bins)


    rebinned_bins, rebinned_projection = rebin(projection[:,1], bins, 39)

    # #plot forward projection
    y_vals = cat(rebinned_bins,rebinned_projection,dims=2)
    plot_vals = y_vals
    #plot!((bins.-1.0), projection,label="projection", color=:green, linetype=:steppost)
    plot!(plot_vals[:,1], plot_vals[:,2],label="projection", color=:green, linetype=:steppost)
    savefig(@sprintf "image_%d" frame)


    return estimate_curve(rebinned_projection, rebinned_bins, angles, head, r, 30 )
end

cwd = @__DIR__
savepath = normpath(joinpath(@__DIR__, "result"))
!isdir(savepath) && mkdir(savepath)
cd(savepath)
for i in 1:27
    plot()
    println(i)
    process_frame(i)
    savefig(@sprintf "angle_test_result_%d" i)
end
#process_frame(21)
#savefig(@sprintf "angle_test_result_%d" i)
