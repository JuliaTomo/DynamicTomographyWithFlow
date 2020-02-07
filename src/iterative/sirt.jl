function _compute_sum_rows_cols(A)
    R_mx1 = spzeros(size(A, 1))
    C_nx1 = spzeros(size(A, 2))

    sum_row = sparse(A * ones( size(A, 2) ) )
    EPS = eps(Float64) # choose float32 to make it more sparse
    R_mx1[sum_row .> EPS] .= 1 ./ (sum_row[sum_row .> EPS])

    sum_col = sparse(A' * ones( size(A, 1) ) )
    C_nx1[sum_col .> EPS] .= 1 ./ (sum_col[sum_col .> EPS])
    
    return R_mx1, C_nx1
end

@doc raw"""
    recon2d_sirt

Reconstruct a 2d image by SIRT

# Args
A : Forward opeartor
b : Projection data 
u0: Initial guess of image
niter: number of iterations

u <- u + CA'R(b - Au)
"""
function recon2d_sirt!(u0::Array{T, 2}, A::SparseMatrixCSC{T,Int}, b0::Array{T, 2}, niter::Int; min_value=nothing, max_value=nothing) where {T <: AbstractFloat}
    R_mx1, C_nx1 = _compute_sum_rows_cols(A)
    
    At = sparse(A') # this significatnly improves the performance
    u = vec(u0)
    b = vec(b0)
    r = similar(b)

    for it = 1:niter
        # println("$it")
        r .= b .- A*u
        u .+= C_nx1 .* (At * (R_mx1 .* r))

        if !isnothing(min_value) # much faster than min_value == nothing
            u .= max.(u, min_value)
        end
        if !isnothing(max_value)
            u .= min.(u, max_value)
        end
        if it % 30 == 0
            residual = sum( r .^ 2 )  / length(r)
            println("$it l2 residual: $residual")
        end
    end
end


# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------

@doc raw"""
    recon2d_slices_sirt

Reconstruct 3d image by SIRT slice by slice

# Args
A : Forward opeartor
b : Projection data 
u0: Initial guess of 3D image [H x W x nslice]
niter: number of iterations

u <- u + CA'R(b - Au)
"""
function recon2d_slices_sirt!(u::Array{T, 3}, A::SparseMatrixCSC{T,Int}, b_::Array{T, 3}, niter::Int; min_value=nothing) where {T <: AbstractFloat}
    if size(b_, 2) != size(u, 3)
        error("Not supported yet when the detector row count == image slice size")
    end

    R_mx1, C_nx1 = _compute_sum_rows_cols(A)
    At = sparse(A') # this significatnly improves the performance
    
    nslice = size(u, 3)

    b_axWxH = permutedims(b_, [1, 3, 2])
    b = reshape(b_axWxH, :, nslice)
    
    halfslice = Int(floor(nslice/2))
    H, W = size(u, 1), size(u, 2)
    
    println("@ preprocessing done. run SIRT slice by slice")
    u_view = reshape(u, :, nslice)

    temp = similar(b)
 
    println("@ num of julia threads: $(Threads.nthreads())")
    # multi threaded version
    for it = 1:niter            
        Threads.@threads for slice=1:nslice
            bb = view(b, :, slice)
            uu = view(u_view, :, slice)
            temp_slice = view(temp, :, slice)
            
            temp_slice .= R_mx1 .* (bb .- mul!(temp_slice, A, uu))
            uu .= C_nx1 .* mul!(uu, At, temp_slice)
        end
        if !isnothing(min_value)
            u .= max.(u, min_value)
        end

        if it % 30 == 0
            # residual = sum( temp .^ 2 )  / length(temp)
            println("iter: $it")
        end
    end
    
    # if no_multi_thread
    # println("MKL library would boost the perforamnce.")
    
    # for it = 1:niter            
    #     temp .= R_mx1 .* (b .- mul!(temp, A, u_view ) )
    #     u_view .= C_nx1 .* mul!(u_view, At, temp)

    #     if !isnothing(min_value)
    #         u .= max.(u, min_value)
    #     end
    # end
    # else   

    println("@ reconstruction done. min: $(minimum(u)), max: $(maximum(u))")
    return u
end
