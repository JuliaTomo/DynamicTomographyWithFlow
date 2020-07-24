using ResumableFunctions
using LsqFit
using Statistics
include("utils.jl")

function get_angle(adjacent, hypotenuse)
    if hypotenuse > adjacent
        a = acos(adjacent/hypotenuse)
        return a
    end
    return 0.0
end

function get_length(adjacent, a)
    return adjacent/cos(a)
end

@resumable function paths(height)
    path = ones(height)
    for i in 1:(2^height)
        @yield path
        last_pos = findlast(p -> p > 0, path)
        if typeof(last_pos) != Nothing
            path[last_pos:end] = -path[last_pos:end]
        end
    end
end

function rebin(projection, bins, s)
    if length(projection) % s != 0
        throw(DomainError("s must divide the number of projections evenly"))
    end

    other_dimension = Int64(length(projection)/s)

    reshaped = reshape(projection, other_dimension, s)
    p = mean!(ones(s)', reshaped)

    reshaped = reshape(bins, other_dimension, s)
    b = mean!(ones(s)', reshaped)
    return b', p'
end

function estimate_curve(projection, bins, angles, head, r, num_points)
    ##################################### MODEL ####################################################
    @. model(x,p) = p[1]*sin(p[2]*x-p[3])+p[4]*sin(p[2]*x-2*p[3]+p[5])

    y1 = 4.0
    y2 = 0.3*y1
    #p = [y1,k,w0,y2,ϕ]
    p0 = [1,0.2,30.0,y2,10.0]
    ################################################################################################
    #estimate angles from projection
    bin_width = bins[4] - bins[3]
    abinned = projection
    angles_est = zeros(length(abinned))
    for i = 1:length(abinned)
        a = get_angle(2*r(0.0), abinned[i])
        angles_est[i] = a
    end

    #estimate length from bins and angle
    length_est = zeros(length(abinned))
    for i = 1:length(abinned)
        l = get_length(bin_width, angles_est[i])
        length_est[i] = l
    end

    #get the bin ends where it seems we're looking at complete tail
    end1, end2 = get_projection_ends(abinned, r)

    angles_est = angles_est[end1:end2]
    length_est = length_est[end1:end2]

    global best_fit = Inf
    global best_curve = zeros(length(angles),2)
    for signs in paths(length(angles_est))
        cks = Array{Complex{Float64},1}(undef,length(angles_est)+1)
        angles_est_signed = angles_est.*signs

        for i = 1:length(angles_est)
            a, l = angles_est_signed[i], length_est[i]
            cks[i+1] = l*ℯ^(im*a)
        end
        cks[1] = head[1]+head[2]*im

        angles_1 = angle.(cks[2:end])
        angles_2 = angle.(cks[1:end-1])

        difference = abs.(angles_1-angles_2)

        if any(d -> d > π/3, difference)
            continue
        end

        xy = get_xy(cks)[2:end,:]

        xdata = xy[:,1]
        ydata = xy[:,2]

        fit = curve_fit(model, xdata, ydata, p0)
        fit_curve = cat(xy[:,1], model(xy[:,1], coef(fit)), dims=2)
        outline = get_outline(fit_curve, r)[1]

        #Define angles and bin  width for forward projection
        projection_fit = parallel_forward(outline,angles,bins)
        end1, end2 = get_projection_ends(projection_fit, r)
        if typeof(end1) != Nothing && typeof(end2) != Nothing
            cost = sum(fit.resid.^2)/length(fit.resid) + sum((projection_fit[end1:end2]-projection[end1:end2]).^2)/length(projection)
            if cost < best_fit
                global best_fit = cost
                global best_curve =xy
            end
        end
    end

    #get curve as fit by model
    fit = curve_fit(model, best_curve[:,1], best_curve[:,2], p0)

    st, en = best_curve[1,1], best_curve[end,1]
    xs = range(st, en, length=num_points)
    xy = cat(xs, model(xs, coef(fit)), dims=2)

    return xy
end
