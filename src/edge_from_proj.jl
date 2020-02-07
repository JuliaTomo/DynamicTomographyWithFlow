"""
Extract blobs from projections
"""

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
