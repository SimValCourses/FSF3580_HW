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

t = ev[1]  # target
figure(1); x = collect(1:n)
for (m,k) in zip(mv,kv)
    global iterror
    outv = complex(zeros(n,k))
    iterror = zeros(n)
    for i in 1:n
        local Q,H,R
        global b,error
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
        iterror[i] = abs(outv[i,1]-t)
    end
    figure(1)
    semilogy(x,iterror)
    figure(k+1)
    scatter(map(x->x.re,ev),map(x->x.im,ev))
    scatter(map(x->x.re,outv[n,:]),map(x->x.im,outv[n,:]),marker="x",color="red")
    title("Setup $m iterations per AM run, restart v build from $k R-v")
end

tol = 1e-10
for (p,k) in zip(mv,kv)
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
    figure(p)
    scatter(map(x->x.re,ee2),map(x->x.im,ee2))
    scatter(map(x->x.re,ee),map(x->x.im,ee),marker="x",color="red")
end
