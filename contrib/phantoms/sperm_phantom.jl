using DelimitedFiles
using Logging
using Plots
using ProgressMeter
using Printf
using Measures
using Dierckx
using PolygonOps
include("../curves/curve_utils.jl")

function rotate_points(list_points::Array{T,2}, ang::T) where T<:AbstractFloat
    rot = [cos(ang) sin(ang); -sin(ang) cos(ang)]
    return list_points*rot
end

function segment_length(a::Array{Float64,1},b::Array{Float64,1})
    value = sqrt((a[1]-b[1])^2+(a[2]-b[2])^2)
    return value
end

function curve_lengths(arr::Array{Float64,2})
    result = Array{Float64,1}(undef,size(arr)[1])
    result[1]=0.0
    for i in 1:size(arr)[1]-1
        result[i+1] = segment_length(arr[i,:],arr[i+1,:])
    end
    csum = cumsum(result)
    return csum
end

#Makes a matrix where the matrix entry is true iff the center of corresponding pixel is not outside the closed curve
#curve is closed curve where first and last point should be the same.
#xs and ys denote the center of each pixel column/rownorm
function closed_curve_to_binary_mat(curve::Array{Float64}, xs::Array{Float64}, ys::Array{Float64})
    N = length(curve[:,1])
    poly = map(i -> SVector{2,Float64}(curve[i,1],curve[i,2]), 1:N)
    result = zeros(Bool, length(xs), length(ys))
    i = 0
    for x in xs
        i += 1
        j = 0#length(ys)+1
        for y in ys
            j +=1
            result[j,i] = inpolygon(SVector(x,y), poly) != 0 ? 1.0 : 0.0
        end
    end
    return result
end

#use smaller grid than for reconstruction to avoid inverse crime
function get_sperm_phantom(nr_frames::Int64; grid_size=0.1)
    cwd = @__DIR__
    cd(cwd)
    #data = readdlm("hyperactive_sperm_trajectory.xyz", '\t')
    data = readdlm("trajectory.xyz", '\t')
    #remove first and last column which artime_sequence[:,1,1]e not relevant
    data = data[1:end, 1:end .!= 1]
    data = data[1:end, 1:end .!= 3]

    #Remove rows that are not numeric
    rows, columns = size(data)

    numeric_rows = filter(i -> all(v->  isa(v, Number), data[i,:]), 1:rows)
    data = data[numeric_rows, :]

    # metrics of dataset
    frames = 3000#1707
    dims = 2
    points = 38#40

    #convert to time sequence (last dimension is frames) - add "head" at 0,0
    time_sequence = zeros(points+1,dims,nr_frames)
    map(t -> time_sequence[2:end,:,t] = data[(t-1)*points+1:(t-1)*points+points,:], 1:nr_frames)

    p = Progress(5,1, "Making phantom")
    r(s) = 1.0
    #Pick every 10th frame to match sampling at synkrotron
    grid = collect(-38.0:grid_size:38.0)
    images = zeros(length(grid),length(grid),nr_frames)
    tracks = Array{Float64,2}[]
    for t=1:nr_frames
        #Remove all zero rows (missing data points)
        non_zeros = filter(i ->  any(v-> v !== 0.0, time_sequence[i,:,t]) ,1:points)
        #prepend!(non_zeros,1)

        center_line = time_sequence[non_zeros,:,t]
        #translate so it starts at zero
        #center_line = center_line.-center_line[1,:]'

        #find angles and rotatet the tail so it is on average parallel with the x-axis
        # this is done by converting points to complex numbers, finding the angles, taking average and turning -average
        cks = get_ck(center_line)

        xy = get_xy(cks)

        angles_est = angle.(cks[2:end])
        average_angle = sum(angles_est)/length(angles_est)
        center_line = rotate_points(xy, -average_angle)
        push!(tracks, center_line)

        #determine outline from skeleton
        outline, normals = get_outline(center_line, r)
        #close the curve
        outline = cat(outline, outline[1,:]', dims=1)
        #convert to binary image
        images[:,:,t] = closed_curve_to_binary_mat(outline,grid,grid)
        next!(p)
    end
    cd(cwd)
    return images, tracks
end
