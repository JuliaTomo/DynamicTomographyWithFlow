function _conv_kern_direct!(out, u, v)
    "Need to refer the source"
    fill!(out, 0)
    u_region = CartesianIndices(u)
    v_region = CartesianIndices(v)
    one_index = oneunit(first(u_region))
    sz2 = size(u, 2)
    for vindex in v_region
        @simd for uindex in u_region
            @inbounds out[uindex + vindex - one_index] += u[uindex] * v[vindex]
        end
    end
    return out
end

# function _conv_kern_direct(
#     u::AbstractArray{T, N}, v::AbstractArray{S, N}, su, sv) where {T, S, N}
#     sout = su .+ sv .- 1
#     out = similar(u, promote_type(T, S), sout)
#     _conv_kern_direct!(out, u, v)
#     return out
# end

"radon transform of LoG"
# $$e_{\sigma}(t)=\left(\frac{1}{\sqrt{2 \pi} \sigma^{5}}\right)\left(t^{2}-\sigma^{2}\right) \exp ^{-t^{2} / 2 \sigma^{2}}$$
# note that we consider 3D

_radon_log(t, sigma; z0) = (t.^2 .- sigma^2 .+ z0^2) .* exp.(-(t.^2 + z0^2 ) / ( 2*sigma^2 ))

"Compute Radon transform of normalized Laplacian of Gaussian in 3D"
function radon_log(q, sigma; z0=0)
    
    nangles, detcount = size(q)
    p_log = zeros(nangles, detcount)
    tt = (-detcount/2+0.5):1.0:(detcount/2-0.5)

    for ang in 1:nangles
        for (i, t) in enumerate(tt)
            dotp = q[ang, :] .* _radon_log.(t .- tt, sigma, z0=z0) ./ ( sqrt(2*pi) * sigma^3 )

            p_log[ang, i] = sum(dotp)
        end
    end
    
    return p_log
end

"""
    radon_filter(p, angles, halfsz, fun_filter)

Radon transform of a filter slice by slice specified by a 1D fucntion `fun_filter`
"""
function radon_filter(p, angles, halfsz, fun_filter)
    nangles, nslice, detcount = size(p)
    p_filtered = similar(p)
    
    kernel=zeros(2*halfsz+1)
    tt = -halfsz:1:halfsz
    out = similar(p)
    p_conv = zeros(2*halfsz+1 + detcount -1)
    
    Threads.@threads for slice=1:nslice
        for ang=1:nangles
            kernel .= fun_filter.(tt, angles[ang])
            _conv_kern_direct!(p_conv, p[ang, slice, :], kernel)
            
            out[ang, slice, :] .= p_conv[halfsz+1:end-halfsz]
        end
    end
    return out
end
