using SparseArrays

function sub2ind(array_shape, rows, cols)
    out = (cols.-1).*array_shape[1] .+ rows
    return convert.(Int64, out)
end

function compute_warping_operator(flow)
    # ref: https://github.com/HendrikMuenster/flexBox/blob/79abc7285703911cca2653434cca5bcefc79c722/operators/generateMatrix/warpingOperator.m
    """
    u: gray image
    v: flow [H, W, 2]
    """
    H, W, dims = size(flow)
    shape = (H,W)
    XX, YY = repeat(collect(1:W)', H, 1), repeat(collect(1:H), 1, W)
    targetPoint2Mat = XX + flow[:,:,1]
    targetPoint1Mat = YY + flow[:,:,2]
    x1 = floor.(targetPoint1Mat).-1
    x2 = x1 .+ 1
    x3 = x1 .+ 2
    x4 = x1 .+ 3
    y1 = floor.(targetPoint2Mat).-1
    y2 = y1 .+ 1
    y3 = y1 .+ 2
    y4 = y1 .+ 3
    v2 = targetPoint1Mat .- x2
    v1 = targetPoint2Mat .- y2
    indicator = ones(H,W)
    indicator = (x1 .> 0).*indicator
    indicator = (x4 .<= W).*indicator
    indicator = (y1 .> 0).*indicator
    indicator = (y4 .<= H).*indicator
    indicator = indicator .> 0
    v1 = v1[indicator]
    v2 = v2[indicator]
    idxList = sub2ind(shape, YY, XX)
    idxList_ = collect(Iterators.flatten(idxList))
    #find the indices
    list_J  = vcat(
        idxList_[sub2ind(shape, x2[indicator], y2[indicator])],
        idxList_[sub2ind(shape, x3[indicator], y2[indicator])],
        idxList_[sub2ind(shape, x2[indicator], y3[indicator])],
        idxList_[sub2ind(shape, x3[indicator], y3[indicator])])
    # list_J  = vcat(
    #     idxList_[sub2ind(shape, x1[indicator], y1[indicator])],
    #     idxList_[sub2ind(shape, x2[indicator], y1[indicator])],
    #     idxList_[sub2ind(shape, x3[indicator], y1[indicator])],
    #     idxList_[sub2ind(shape, x4[indicator], y1[indicator])],
    #     idxList_[sub2ind(shape, x1[indicator], y2[indicator])],
    #     idxList_[sub2ind(shape, x2[indicator], y2[indicator])],
    #     idxList_[sub2ind(shape, x3[indicator], y2[indicator])],
    #     idxList_[sub2ind(shape, x4[indicator], y2[indicator])],
    #     idxList_[sub2ind(shape, x1[indicator], y3[indicator])],
    #     idxList_[sub2ind(shape, x2[indicator], y3[indicator])],
    #     idxList_[sub2ind(shape, x3[indicator], y3[indicator])],
    #     idxList_[sub2ind(shape, x4[indicator], y3[indicator])],
    #     idxList_[sub2ind(shape, x1[indicator], y4[indicator])],
    #     idxList_[sub2ind(shape, x2[indicator], y4[indicator])],
    #     idxList_[sub2ind(shape, x3[indicator], y4[indicator])],
    #     idxList_[sub2ind(shape, x4[indicator], y4[indicator])])
    list_J = collect(Iterators.flatten(list_J))
    list_val  = ( (1 .-v1) .* (1 .-v2))
    list_val = vcat(list_val,(v2 .* (1 .-v1)))
    list_val = vcat(list_val,(v1 .* (1 .-v2)))
    list_val = vcat(list_val,(v1 .* v2))
    # list_val =vcat(
    #     (v1.*v2.*(v1 .- 1).^2 .*(v2 .- 1).^2)/4,
    #     -1 .*(v1.*(v1 .- 1).^2 .*(3 .*v2.^3 .- 5 .*v2.^2 .+ 2))./4,
    #     -1 .*(v1.*v2.*(v1 .- 1).^2 .*(- 3 .*v2.^2 + 4 .*v2 .+ 1))./4,
    #     -1 .*(v1.*v2.^2 .*(v1 .- 1).^2 .*(v2 .- 1))./4,
    #     -1 .*(v2.*(v2 .- 1).^2 .*(3 .*v1.^3 .- 5 .*v1.^2 .+ 2))./4,
    #     ((3 .*v1.^3 .- 5 .*v1.^2 .+ 2).*(3 .*v2.^3 .- 5 .*v2.^2 .+ 2))./4,
    #     (v2.*(- 3 .*v2.^2 .+ 4 .*v2 .+ 1).*(3 .*v1.^3 .- 5 .*v1.^2 .+ 2))./4,
    #     (v2.^2 .*(v2 .- 1).*(3 .*v1.^3 .- 5 .*v1.^2 .+ 2))./4,
    #     -1 .*(v1.*v2.*(v2 .- 1).^2 .*(- 3 .*v1.^2 .+ 4 .*v1 .+ 1))./4,
    #     (v1.*(- 3 .*v1.^2 .+ 4 .*v1 .+ 1).*(3 .*v2.^3 .- 5 .*v2.^2 .+ 2))./4,
    #     (v1.*v2.*(- 3 .*v1.^2 .+ 4 .*v1 .+ 1).*(- 3 .*v2.^2 .+ 4 .*v2 .+ 1))./4,
    #     (v1.*v2.^2 .*(v2 .- 1).*(- 3 .*v1.^2 .+ 4 .*v1 .+ 1))./4,
    #     -1 .*(v1.^2 .*v2.*(v1 .- 1).*(v2 .- 1).^2)./4,
    #     (v1.^2 .*(v1 .- 1).*(3 .*v2.^3 .- 5 .*v2.^2 .+ 2))./4,
    #     (v1.^2 .*v2.*(v1 .- 1).*(- 3 .*v2.^2 .+ 4 .*v2 .+ 1))./4,
    #     (v1.^2 .*v2.^2 .*(v1 .- 1).*(v2 .- 1))./4
    # )
    list_val = collect(Iterators.flatten(list_val))
    list_I = idxList[indicator]
    #list_I =collect(Iterators.flatten(repeat(list_I,1,16)))
    list_I =collect(Iterators.flatten(repeat(list_I,1,4)))
    # list_I_py, list_J_py, list_val_py = py"py_warping_operator"(flow)
    # list_I_py = list_I_py.+1
    # list_J_py = list_J_py.+1
    # W_mat_py = sparse(list_I_py, list_J_py, list_val_py, H*W, H*W)
    W_mat = dropzeros!(sparse(list_I, list_J, list_val, H*W, H*W))
    return W_mat
end

img2 = zeros(128,128)
H, W = size(img2)
img2[30:60,40:50] .= 255.0

flow = zeros(H, W, 2)
flow[30:60,40:50, 1] .= -1.0

Wop = compute_warping_operator(flow)
    img1_ = Wop * vec(img2)
img1 = reshape(img1_, H, W)
