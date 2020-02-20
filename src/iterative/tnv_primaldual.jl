using .util_convexopt

function _recon2d_tnv_primaldual!(u::Array{T, 2, 2}, A, b0::Array{T, 2, 2}, niter, w_tnv, sigmas, tau)
    At = sparse(A')
    H, W, C = size(u)
    
    b = reshape(b0, (size(b0, 1)*size(b0, 2), C))
    # b = vec(b0)
    
    ubar = deepcopy(u)
    u_prev = similar(u)

    p1 = zeros(size(b))
    p2 = zeros(H, W, 2, C)

    p_adjoint = zeros(H, W, C)

    du = similar(p2)
    divp2 = similar(p2)
    
    data1 = similar(b)

    for it=1:niter
        u_prev .= u

        for c=1:C
            ubar_c = view(ubar, :, :, c)
            data1_c = view(data1, :, c)
            p_adjoint_c = vec(view(p_adjoint, :, :, c))

            # dual update: data fidelity
            data1_c .= mul!(data1_c, A, vec(ubar_c)) .- view(b, :, c)
            p1[:,c] .= (view(p1, :, c) .+ sigmas[1] * data1_c) ./ (sigmas[1] + 1.0) # l2 norm
            mul!(p_adjoint_c, At, view(p1, :, c))
            
            # dual update: TNV
            du_c = view(du, :, :, :, c)            
            grad!(du_c, ubar_c)
            p2[:,:,:,c] .+= sigmas[2] .* du_c
            
            # projection onto dual of (S1,l1)
            
            util_convexopt.proj_dual_S1l1!(view(p2,:,:,:,c), w_tnv) #anisotropic TV
            
            p_adjoint_c .-= div!(view(divp2,:,:,:,c), p2) # p1_adjoint + p2_adjoint
        end

        # primal update
        u .= max.(u .- tau .* p_adjoint, 0.0) # positivity constraint

        # acceleration
        ubar .= 2 .* u .- u_prev

        # compute primal energy (optional)
        if it % 50 == 0
            energy = sum(data1.^2) / length(data1) + sum(abs.(du)) / length(du)
            println("$it, approx. primal energy: $energy")
        end
    end
    return u
end

"""
    recon2d_tv_primaldual!(u::Array{T, 2}, A, b::Array{T, 2}, niter::Int, w_tv::T, c=1.0)

Reconstruct a 2d image by TV-L2 model using Primal Dual optimization method

# Args
u : Initial guess of images
A : Forward opeartor
b : Projection data 
niter: number of iterations
w_tv: weight for TV term
c : See 61 page in 2016_Chambolle,Pock_An_introduction_to_continuous_optimization_for_imagingActa_Numerica
"""
function recon2d_tnv_primaldual!(u::Array{T, 2, 2}, A, b::Array{T, 2, 2}, niter::Int, w_tnv::T, c=1.0) where {T <: AbstractFloat}
    if size(u, 3) != size(b, 3)
        error("The channel size of u and b should match.")
    end

    @time op_A_norm = util_convexopt.compute_opnorm(A)
    println("@ opnorm of forward projection operator: $op_A_norm")
    ops_norm = [op_A_norm, sqrt(8)]
    
    sigmas = zeros(length(ops_norm))
    for i=1:length(ops_norm)
        sigmas[i] = 1.0 / (ops_norm[i] * c)
    end

    tau = c / sum(ops_norm)
    println("@ step sizes sigmas: ", sigmas, ", tau: $tau")
    
    return _recon2d_tnv_primaldual!(u, A, b, niter, w_tnv, sigmas, tau)
end

