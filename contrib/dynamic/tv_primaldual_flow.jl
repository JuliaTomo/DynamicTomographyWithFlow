

function _recon2d_tv_primaldual_flow(As,A_norm,W_list,W_norm,u0s,bs,w_tv,w_flow,c,niter)
    #A variational reconstruction method for undersampled dynamic x-ray tomography based on physical motion models (Burger)#
    height, width, frames = size(u0s)

    u = u0s
    ubar = deepcopy(u)
    #
    p1 = zeros(size(bs))
    p2 = zeros(height, width, 2,frames)
    p3 = zeros(height, width, frames)


    p_adjoint  = similar(p3)
    p3_adjoint = similar(p3)
    opsnorm = [A_norm, sqrt(8), W_norm]
    tau = c /sum(opsnorm)
    for it=1:niter
        u_prev = deepcopy(u)
        p_adjoint_prev = deepcopy(p_adjoint)
        for t=1:frames
            A = As[t]
            _p1, _p2, _p3, _ubar, _p_adjoint, _p3_adjoint = view(p1,:,t), view(p2,:,:,:,t), view(p3,:,:,t), view(ubar,:,:,t), view(p_adjoint,:,:,t), view(p3_adjoint,:,:,t)
            sigmas = map(n-> 1.0/(n*c), opsnorm)

            ops = [A,D]
            data1, data2, p1_adjoint, p2_adjoint = get_tv_adjoints!(ops, bs[:,t], ubar[:,:,t], w_tv, sigmas[1:2], _p1, _p2, height, width)
            p_adjoint[:,:,t] = p1_adjoint + p2_adjoint

            if t < frames
                _ubar_1 = view(ubar,:,:,t+1)
                Wu = W_list[t]*(collect(Iterators.flatten(_ubar_1))) - (collect(Iterators.flatten(_ubar)))
                p3_ascent = _p3 + sigmas[3] * reshape(Wu, height, width)
                _p3[:,:] = proj_dual_l1(p3_ascent, w_flow)

                p3_adjoint_t1 = W_list[t]'*vec(_p3)
                _p3_adjoint[:,:] += reshape(p3_adjoint_t1, height,width)
                _p3_adjoint[:,:] += -_p3
            end
            p_adjoint[:,:,t] += p3_adjoint[:,:,t]
        end

        # primal update
        u_descent = u - tau*p_adjoint
        u = max.(u_descent, 0.0) # positivity constraint
        du = u_prev - u
        primal_gap = mean(abs.(-p_adjoint+p_adjoint_prev + du/tau))
        # acceleration
        ubar = 2*u - u_prev
        @info "primal gap:" primal_gap
    end
    return u
end
