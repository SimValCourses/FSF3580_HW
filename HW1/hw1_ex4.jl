using LinearAlgebra, Random, MatrixDepot, PyPlot
include("GS.jl")
include("arnoldi.jl")

nn = 100
Random.seed!(0)
A = matrixdepot("wathen",nn,nn)
n = size(A)[1]
b = randn(n)

mmax = 80
Km = zeros(n,mmax)
Km[:,1] = b/norm(b);
λAM = zeros(mmax,mmax)
λGM = zeros(mmax,mmax)

for m in 1:mmax
    local Q,Hb,H
    # build Krylov space
    if m > 1
        v = A*Km[:,m-1];
        Km[:,m] = v/norm(v);
        # min angle
        display(acos(maximum(abs.(Km[:,m]'*Km[:,1:m-1])))*180/pi)
    end
    # Km^TKm
    KTK = transpose(Km[:,1:m])*Km[:,1:m]
    # Km^T A Km
    KTAK = transpose(Km[:,1:m])*A*Km[:,1:m]
    # Solve generalized EVP
    Fg = eigen(KTAK,KTK)
    # find largest EigVal
    I = sortperm(abs.(Fg.values),rev=true)
    λGM[m,1:m] = real(Fg.values[I])

    # arnoldi
    Q,Hb = arnoldi(A,b,m,3)
    # extract Hm
    H = Hb[1:m,:]
    # find Ritz pair
    Fa = eigen(H)
    # find largest Eigval
    I = sortperm(abs.(Fa.values),rev=true)
    λAM[m,1:m] = real(Fa.values[I])
end
figure(1)
x = 1:mmax
# (2)
for m in 1:mmax
    m==1 ? plot(x[m:mmax],λGM[m:mmax,m],color="r",label="(2)") : plot(x[m:mmax],λGM[m:mmax,m],color="r")

end
# AM
for m in 1:mmax
    m==1 ? plot(x[m:mmax],λAM[m:mmax,m],color="b",label="AM") : plot(x[m:mmax],λAM[m:mmax,m],color="b")
end
xlabel("Krylov space dimension")
ylabel("Re(λ_r)")
ylim(0,400)
legend()
return
