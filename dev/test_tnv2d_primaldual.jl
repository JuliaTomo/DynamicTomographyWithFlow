using TomoForward
using XfromProjections

include("../src/iterative/tnv_primaldual.jl")

# img = imread("test_data/shepplogan512.png")[:,:,1]
# H, W = 128, 128
# img = imresize(img, H, W)

img1 = zeros(128, 128)
img1[40:60, 40:60] .= 1.0
H, W = size(img1)

img2 = zeros(H, W)
img2[40:60, 40:60] .= 2.0

src_origin = 10.0
det_origin = 20.0

nangles = 30
detcount = Int(floor(size(img1,1)*1.4))
proj_geom = ProjGeomFan(1.0, detcount, LinRange(0,pi,nangles+1)[1:nangles], src_origin, det_origin)

# test line projection model
A = fp_op_fan_line(proj_geom, size(img, 1), size(img, 2))
p1 = A * vec(img1)
p = reshape(p, nangles, detcount)

u0 = zeros(size(img))
niter=500
lambdas = [0.01, 0.1, 0.6]
u = zeros(H, W, 5)
u[:,:,1] = recon2d_tv_primaldual!(u0, A, p, niter, lambdas[3], 0.01)
u[:,:,2] = recon2d_tv_primaldual!(u0, A, p, niter, lambdas[3], 10)

for (i,lamb) in enumerate(lambdas)
    w_tv=lamb
    c=1.0

    u[:,:,i+2] = recon2d_tv_primaldual!(u0, A, p, niter, w_tv, c)
end

if false
    using PyPlot
    ax00 = plt.subplot2grid((2,3), (0,0))
    ax01 = plt.subplot2grid((2,3), (0,1))
    ax02 = plt.subplot2grid((2,3), (0,2))
    ax10 = plt.subplot2grid((2,3), (1,0))
    ax11 = plt.subplot2grid((2,3), (1,1))
    ax12 = plt.subplot2grid((2,3), (1,2))
    ax00.imshow(img); ax00.set_title("original")
    ax01.imshow(u[:,:,1]); ax01.set_title("lamb $(lambdas[3]) c=0.01")
    ax02.imshow(u[:,:,2]); ax02.set_title("lamb $(lambdas[3]) c=10")
    ax10.imshow(u[:,:,3]); ax10.set_title("lamb $(lambdas[1]) c=1")
    ax11.imshow(u[:,:,4]); ax11.set_title("lamb $(lambdas[2]) c=1")
    ax12.imshow(u[:,:,5]); ax12.set_title("lamb $(lambdas[3]) c=1")
    suptitle("original and reconstruction results")
    show()
end