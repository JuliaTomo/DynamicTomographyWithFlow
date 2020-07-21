#using PyCall
using Logging
using Dierckx
using IterTools
#using .snake_forward
#using .curve_utils
using Plots
using MATLAB
using PyCall
using EMST
using DataFrames
using CurveProximityQueries
using IntervalArithmetic

using Luxor

include("./snake_forward.jl")
include("./curve_utils.jl")
#Reimplementation of Vedranas method with modifications.

#VEDRANA MODIFIED
function displace(centerline_points, force, radius_func, w, w_u, w_l; plot=false)
    L = size(centerline_points,1)
    (outline_xy, normal) = get_outline(centerline_points, radius_func)

    outline_normals = snake_normals(outline_xy)
    forces = force.*outline_normals

    mid = Int64(size(outline_xy,1)/2)#always even
    upper_forces = forces[1:mid,:]
    lower_forces = (forces[mid+1:end, :])[end:-1:1,:]

    upper_points = outline_xy[1:mid,:]
    lower_points = (outline_xy[mid+1:end, :])[end:-1:1,:]

    displaced_upper_points = upper_points .+ w*(upper_forces.*w_u);
    displaced_lower_points = lower_points .+ w*(lower_forces.*w_l);
    displaced_centerline = zeros(L,2)

    displaced_centerline[2:end-1,:] = (displaced_upper_points[3:end-2,:]+displaced_lower_points[3:end-2,:])./2
    displaced_centerline[1,:] = (displaced_upper_points[1,:]+displaced_lower_points[1,:]+displaced_upper_points[2,:]+displaced_lower_points[2,:])./4
    displaced_centerline[L,:] = (displaced_upper_points[end,:]+displaced_lower_points[end,:]+displaced_upper_points[end-1,:]+displaced_lower_points[end-1,:])./4
    if plot
        f = cat(w*(upper_forces.*w_u), (w*(lower_forces.*w_l))[end:-1:1,:], dims = 1).*25
        quiver!(outline_xy[:,1], outline_xy[:,2], quiver=(f[:,1], f[:,2]), color=:gray)
    end
    return displaced_centerline
end

function move_points(residual,curves,angles,N,centerline_points,r,w,w_u, w_l; plot=false)
    (x_length, y_length) = size(residual)
    vals = zeros(Float64, N)
    if y_length > 1
        F = Spline2D(collect(1:1.0:x_length), collect(1:1.0:y_length), residual, kx=1, ky=1);
        vals = zeros(Float64, N)
        for i = 1:length(angles)
            interp = F(curves[:,i], repeat([i], N))
            vals += interp
        end
    else
        F = Spline1D(collect(1:1.0:x_length), residual[:,1], k=1);
        interp = F(curves[:,1])
        vals = interp
    end

    force = vals*(1/length(angles))

    centerline_points = displace(centerline_points, force, r, w, w_u, w_l, plot=plot)
    return centerline_points
end

function to_pixel_coordinates(current, angles, bins)
    N = size(current,1);
    vertex_coordinates = zeros(Float64,N,length(angles));
    a = (length(bins)-1)/(bins[end]-bins[1]); # slope
    b = 1-a*bins[1]; # intercept
    for k = 1:length(angles)
        angle = angles[k]
        projection = [cos(angle) sin(angle)]';
        #expressing vertex coordinates as coordinates in sinogram (pixel coordinates, not spatial)
        vertex_coordinates[:,k] = (current*projection)*a.+b;
    end
    return vertex_coordinates
end

function evolve_curve(sinogram_target, centerline_points, r, angles, bins, max_iter, w, w_u, w_l, smoothness, degree::Int64; plot=false)
    (current, normal) = get_outline(centerline_points, r)
    current_sinogram = parallel_forward(current,angles,bins)

    curves = to_pixel_coordinates(current, angles, bins);
    mu = sum(sinogram_target[:].*current_sinogram[:])/sum(current_sinogram[:].^2)
    residual = sinogram_target - mu*current_sinogram
    N = size(current,1)
    centerline_start = centerline_points[1,:]
    for iter  = 1:max_iter
        centerline_points = move_points(residual,curves,angles,N,centerline_points,r, w, w_u,w_l, plot=plot)

        L = size(centerline_points,1)
        #HACK
        cp = eliminate_loopy_stuff(centerline_points, 2*r(0.0))
        #HACK
        if size(cp,1) > degree
            centerline_points = cp
        else
            @warn "Too loopy"
        end
        t = curve_lengths(centerline_points)

        spl = ParametricSpline(t,centerline_points',k=degree, s=0.0)
        #HACK
        if smoothness > 0.0
            try
                spl = ParametricSpline(t,centerline_points',k=degree, s=smoothness)
            catch e
                @warn e
            end
        end

        tspl = range(0, t[end], length=L)
        centerline_points = collect(spl(tspl)')
        centerline_points[1,:] = centerline_start
        (current, normal) = get_outline(centerline_points, r)
        current_sinogram = parallel_forward(current,angles,bins);
        curves = to_pixel_coordinates(current, angles, bins);
        mu = sum(sinogram_target[:].*current_sinogram[:])/sum(current_sinogram[:].^2)
        residual = sinogram_target - mu*current_sinogram;
    end
    return centerline_points
end

function recon2d_tail(centerline_points::AbstractArray{T,2}, r, angles::Array{T},bins::Array{T},sinogram_target::Array{T,2}, max_iter::Int, smoothness::T, w::T, degree::Int64, w_u::Array{T}, w_l::Array{T}; plot=false) where T<:AbstractFloat
    current = evolve_curve(sinogram_target, centerline_points, r, angles, bins, max_iter, w, w_u, w_l, smoothness, degree, plot=plot)
    return current
end


#find skeleton using skimage
function get_skeleton(img)
    #skeletonize it using skimage
    morphology = pyimport("skimage.morphology")
    skeleton = morphology.skeletonize(img)
    return skeleton
end

#TODO PUT IN COMMON PLACE
function closed_curve_to_binary_mat_2(curve::Array{Float64}, xs::Array{Float64}, ys::Array{Float64})::Array{Bool,2}
    N = length(curve[:,1])
    poly = map(i -> SVector{2,Float64}(curve[i,1],curve[i,2]), 1:N)
    result = zeros(Bool, length(xs), length(ys))
    i = 0
    for x in xs
        i += 1
        j = 0#length(ys)+1
        for y in ys
            j +=1
            result[j,i] = inpolygon(SVector(x,y), poly) != 0 ? true : false
        end
    end
    println(typeof(result))
    return result
end

#newtons root finding method
function newton(f::Function, x0::Number, fprime::Function, args::Tuple=();
                tol::AbstractFloat=1e-8, maxiter::Integer=500, eps::AbstractFloat=1e-10)
    for _ in 1:maxiter
        yprime = fprime(x0, args...)
        if abs(yprime) < eps
            @warn "First derivative is zero"
            return x0
        end
        y = f(x0, args...)
        x1 = x0 - y/yprime
        if abs(x1-x0) < tol
            return x1
        end
        x0 = x1
    end
    @warn "Max iteration exceeded"
    return x0
end

#determine intersection of circle with center (a,b) and rdius r and a parametric spline curve starting at parameter s0. Uses newtons method to determine root.
function closests_spline_point(curve, p;s0=0.0)
    dist_square(s) = (curve(s)[1]-p[1])^2+(curve(s)[2]-p[2])^2

    println(dist_square(s0))
    derr(s) = derivative(curve,[s])
    println(derr(s0))
    println(curve(s0))
    dist_prime(s) = 2*derr(s)[1]-2*derr(s)[1]*p[1]+2*derr(s)[2]-2*derr(s)[2]*p[2]
    println(dist_prime(s0))
    s0 = newton(dist_square,s0,dist_prime)
    return s0
end

#Get the ends of the cell midcurve. The one with most 1 pixels round it is the head.
function find_head(possibilities::Array{Array{CartesianIndex,N} where N,1}, img::Array{Bool,2})::Array{Array{CartesianIndex,N} where N,1}
    #find the possibility which has most surrounding pixels
    height,width = size(img)
    r = 20
    possibilities = sort(possibilities, by=x->get_disk_intersection(img,x[1][2],x[1][1],r)[1],rev=true)
    return possibilities
end

#Given an array of cartesian indexes get the lengths between consecutive points
function get_lengths(arr::Array{CartesianIndex{2},1})
    result = Array{Float64,1}(undef,length(arr))
    result[1]=0.0
    for i in 1:length(arr)-1
        result[i+1] = cartesian_2_norm(arr[i],arr[i+1])
    end
    csum = cumsum(result)
    return csum
end

#Given an array of float tuples get the lengths between consecutive points
function get_lengths(arr::Array{Tuple{Float64,Float64},1})
    result = Array{Float64,1}(undef,length(arr))
    result[1]=0.0
    for i in 1:length(arr)-1
        result[i+1] = cartesian_2_norm(arr[i],arr[i+1])
    end
    csum = cumsum(result)
    return csum
end

#Sorts the "spine" of the cell from head to tail pixel
function sort_pixels(arr::Array{Array{CartesianIndex,N} where N,1}, img)::Array{CartesianIndex{2},1}
    ends = find_head(get_ends(arr),img)
    current = ends[1]
    result = Array{CartesianIndex,1}(undef,length(arr))
    i = 1
    while length(arr) > 0
        println(length(arr))
        filter!(x -> x != current, arr)
        result[i] = current[1]#first element is itself, rest are neighbors
        next = get_next(current,arr)
        current = next
        i = i+1
    end
    return result
end

function neighbors(vertex_index, edge_list)
    first = findall(i -> edge_list[i,2] == vertex_index, 1:size(edge_list,1))
    second = findall(i -> edge_list[i,1] == vertex_index, 1:size(edge_list,1))
    return cat(edge_list[first,1], edge_list[second,2], dims = 1)
end
#
function degree(vertex_index, edge_list)
    return length(neighbors(vertex_index, edge_list))
end

function remove_vertex(vertex_index, edge_list)
    return edge_list[findall(i -> edge_list[i,2] != vertex_index && edge_list[i,1] != vertex_index, 1:size(edge_list,1)), :]
end

function get_path(root, edge_list)
    path = [root]
    current_vertex = root
    deg = degree(current_vertex, edge_list)
    while deg == 1
        n = neighbors(current_vertex, edge_list)
        append!(path,n)
        edge_list = remove_vertex(current_vertex, edge_list)
        current_vertex = n[1]
        deg = degree(current_vertex, edge_list)
    end
    return path
end

function extend_centerline(centerline, L)
    current_length = curve_lengths(centerline)[end]
    if current_length < L
        extra_point = [11.73528125, -32.3412619444]#centerline[end,:].+(L-current_length)*(last_direction./norm(last_direction))
        centerline = cat(centerline, extra_point', dims = 1)
    end
    return centerline
end

meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

function force(p, head, tail, b, radius_func, outline)
    stat_point = @SVector [p[i] for i in 1:2]
    closest = closest_points(stat_point,b)
    f_head = head.-p
    f_tail = tail.-p
    if typeof(closest) != Nothing
    #
        #vector going from p to closest centerline point
        f_closest = p.-closest[2]


    #
    #     #If the point is closer to the centerline than the radius, then repel from centerline
        poly = map(i -> Point(outline[i,1],outline[i,2]), 1:size(outline,1))
        if norm(f_closest) < radius_func(0.0) && isinside(Point(closest[2][1], closest[2][2]), poly)
            return f_closest
        end

    end
    return 1/norm(f_tail)*f_tail+1/norm(f_head)*f_head
end

function force_2(outline, head)
    forces = zeros(size(outline))
    forces[1,:] = head .- outline
    N = size(outline,1)/2

function find_closest_point(p,curve)
    closest_point_idx = 1
    for i = 2:size(curve,1)
        if norm(p-curve[closest_point_idx,:]) > norm(p-curve[i,:])
            closest_point = i
        end
    end
    return i
end


function get_centerline(outline, head, radius_func, L)
    # #TODO MAKE THIS INPUT ARGUMENT
    # #close the curve
    # outline_length = size(outline,1)
    # stat_point = @SVector [head[i] for i in 1:2]
    # b_outline = Bernstein(map(i -> path_coords[i,:], 1:size(path_coords,1)), limits=Interval(0.0, 1.0))
    # closest = closest_points(stat_point,b_outline)
    #Make head first point on the outline
    closest_to_head = find_closest_point(head, outline)
    circshift!(outline, -(closest_to_head-1))

    outline_closed = cat(outline, outline[1,:]', dims=1)
    grid = collect(-100.0:0.5:100.0)
    img = closed_curve_to_binary_mat_2(outline_closed,grid,grid)
    skeleton = get_skeleton(img)
    indices = findall(skeleton)
    coordinates = collect(Iterators.product(grid,grid))[indices]
    euclidean_coords = cat(last.(coordinates)',first.(coordinates)', dims=1)
    euclidean_coords = cat(head, euclidean_coords, dims=2)
    try
        e_test  = compute_emst(euclidean_coords);
        vertex_indices = 1:size(euclidean_coords,2);
        path_from_head = get_path(1,e_test)
        path_coords = (euclidean_coords[:,path_from_head])'
        path_coords = extend_centerline(path_coords, L)
        #
        t = curve_lengths(path_coords)
        #
        b = Bernstein(map(i -> path_coords[i,:], 1:size(path_coords,1)), limits=Interval(0.0, t[end]))
        points = b.(collect(0.0:0.01:t[end]))
        #
        plot!(first.(points), last.(points), label = "bernstein")
        # tail = path_coords[end,:]
        forces = zeros(size(outline))
        tail = [11.73528125, -32.3412619444]
        for j in 1:size(outline,1)

            forces[j,:] = force(outline[j,:], head, tail, b, radius_func, outline)

        end

        x, y = outline[:,1],outline[:,2]
        f(x,y) = force([x,y],head,tail,b,radius_func)#, outline)
        uv = f.(x,y)
        quiver!(x,y,quiver=(first.(uv),last.(uv)))
        new_outline = outline.+0.2*forces
        return new_outline
    catch
        return outline
    end

    return outline
end

function regularization_matrix(N,alpha,beta)
    cwd = @__DIR__
    B = mat"addpath($cwd); regularization_matrix($N,$alpha,$beta);"
    return B
end
#
function remove_crossings(curve)
    cwd = @__DIR__
    curve = mat"addpath($cwd); remove_crossings($curve);"
    return curve
end
#
function distribute_points(curve)
    curve = cat(dims = 1, curve, curve[1,:]'); # closing the curve
    N = size(curve,1); # number of points [+ 1, due to closing]
    dist = sqrt.(sum(diff(curve, dims=1).^2, dims=2))[:,1]; # edge segment lengths
    t = prepend!(cumsum(dist, dims=1)[:,1],0.0) # total curve length

    tq = range(0,t[end],length=N); # equidistant positions
    curve_new_1 = Spline1D(t,curve[:,1], k=1).(tq); # distributed x
    curve_new_2 = Spline1D(t,curve[:,2], k=1).(tq); # distributed y
    curve_new = hcat(curve_new_1,curve_new_2); # opening the curve again
    return curve_new[1:end-1,:]
end

function move_points(residual,curves,angles,N,centerline_points,r,w,w_u, w_l; plot=false)
    (x_length, y_length) = size(residual)
    vals = zeros(Float64, N)
    if y_length > 1
        F = Spline2D(collect(1:1.0:x_length), collect(1:1.0:y_length), residual, kx=1, ky=1);
        vals = zeros(Float64, N)
        for i = 1:length(angles)
            interp = F(curves[:,i], repeat([i], N))
            vals += interp
        end
    else
        F = Spline1D(collect(1:1.0:x_length), residual[:,1], k=1);
        interp = F(curves[:,1])
        vals = interp
    end

    force = vals*(1/length(angles))

    centerline_points = displace(centerline_points, force, r, w, w_u, w_l, plot=plot)
    return centerline_points
end

function move_points_original(residual,curves,angles,N,current,B,w, head, r, L)
    (x_length, y_length) = size(residual)
    vals = zeros(Float64, N)
    if y_length > 1
        F = Spline2D(collect(1:1.0:x_length), collect(1:1.0:y_length), residual, kx=1, ky=1);
        vals = zeros(Float64, N)
        for i = 1:length(angles)
            interp = F(curves[:,i], repeat([i], N))
            vals += interp
        end
    else
        F = Spline1D(collect(1:1.0:x_length), residual[:,1], k=1);
        interp = F(curves[:,1])
        vals = interp
    end
    force = (w.*vals)*(1/length(angles))
    normals = snake_normals(current)
    vectors = force.*normals
    current = current + vectors;
    plot!(current[:,1],current[:,2], label="outline")
    current = get_centerline(current, head, r, L)
    current = distribute_points(remove_crossings(B*current))
    plot!(current[:,1],current[:,2], label="nexw_outline")
    return current
end

function evolve_curve_original(sinogram_target, current, angles, bins, B, max_iter, w, head, r, L)
    current_sinogram = parallel_forward(current,angles,bins)

    curves = to_pixel_coordinates(current, angles, bins);

    mu = sum(sinogram_target[:].*current_sinogram[:])/sum(current_sinogram[:].^2)
    residual = sinogram_target - mu*current_sinogram
    N = size(current,1)

    for iter  = 1:max_iter
        current = move_points_original(residual,curves,angles,N,current,B,w,head,r, L);
        current_sinogram = parallel_forward(current,angles,bins);
        curves = to_pixel_coordinates(current, angles, bins);
        mu = sum(sinogram_target[:].*current_sinogram[:])/sum(current_sinogram[:].^2)
        residual = sinogram_target - mu*current_sinogram;
    end
    return current
end

function vedrana(current::Array{T,2},angles::Array{T},bins::Array{T},sinogram_target::Array{T,2}, max_iter::Int, alpha::T, beta::T, w::Array{T}, head, r, L) where T<:AbstractFloat
    N = size(current,1)
    B = regularization_matrix(N,alpha,beta)
    current = evolve_curve_original(sinogram_target, current, angles, bins, B, max_iter, w, head, r, L)
    return current
end
#
