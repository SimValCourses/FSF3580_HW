using LinearAlgebra, MAT, PyPlot, Statistics, Random
include("arnoldi.jl")
include("arnupd.jl")
include("arnoldi_sorensen.jl")
include("GS.jl")
Random.seed!(0)
bw = MAT.matread("Bwedge.mat")

ev = bw["B_eigvals"];
A = bw["B"];
N = length(ev)

b = randn(N)    # starting vector
mv = [ 10, 20 ] # number of iterations
kv = [ 5, 10 ]  # basis for new starting vector
n = 100         # number of restarts
v = 3           # GS scheme (3: DCGS)
#=
t = ev[1]  # target
figure(1); x = collect(1:n)
for (l,(m,k)) in enumerate(zip(mv,kv))
    outv = complex(zeros(n,k))
    for i in 1:n
        local Q,H,R
        global b
        i==1 ? println("Startup") : println("Restart $(i-1)")
        Q,H = arnoldi(A,b,m,v)
        # extract Ritz values
        R = eigen(H[1:m,1:m]);
        ridx = partialsortperm(abs.(R.values),1:k,rev=true)
        outv[i,:] = R.values[ridx]
        Rv = Q[:,1:m]*R.vectors[:,ridx]
        # construct restart vector
        weights = ones(k,1)
        b = Rv*weights
    end
    figure(1)
    subplot(2,2,l)
    for i = 1:k
        scatter(x,real(outv[:,i]),10)
    end
    l==1 ? ylabel("Re(eig)") : nothing
    title("m=$m, k=$k")

    subplot(2,2,l+2)
    for i = 1:k
        scatter(x,imag(outv[:,i]),10)
    end
    xlabel("Restart count")
    l==1 ? ylabel("Im(eig)") : nothing

    figure(2)
    subplot(1,2,l)
    scatter(real(ev),imag(ev))
    scatter(real(outv[n,:]),imag(outv[n,:]),marker="x",color="r")
    l==1 ? ylabel("Im") : nothing
    xlabel("Re")
    title("m=$m, k=$k")
    plt.gcf().gca().set_aspect("equal")
    #title("Setup $m iterations per AM run, restart v build from $k R-v")
end
=#

tol = 1e-10
figure(3)
for (i,(p,k)) in enumerate(zip(mv,kv))
    global tol, A
    local n
    n=length(ev)
    v1=ones(n);
    v1=v1/norm(v1);
    V,H,r=arnupd(A,k,p,tol,v1);
    ee=eigvals(H[1:k,1:k]);
    ee2=ev
    #I2=sortperm(vec(abs.(ee2)),rev=true); ee2=ee2[I2]; ee2=ee2[1:k];
    I1=sortperm(vec(abs.(ee)),rev=true);  ee=ee[I1];
    subplot(1,2,i)
    scatter(real(ee2),imag(ee2))
    scatter(real(ee),imag(ee),marker="x",color="r")
    plt.gcf().gca().set_aspect("equal")
    i==1 ? ylabel("Im") : nothing
    xlabel("Re")
    title("m=$p, k=$k")
end
