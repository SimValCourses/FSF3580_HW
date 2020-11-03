using LinearAlgebra, Random, MatrixDepot, PyPlot
include("GS.jl")
include("arnoldi.jl")

n = 100
Random.seed!(0)
A = randn(n,n)
b = randn(n)

mmax = 80
Km = zeros(n,mmax)
λ = zeros(mmax,2)
#figure()
for m in 1:mmax
    println("Step $m:")
    # build Krylov space
    v = A^(m-1)*b;
    Km[:,m] = v/norm(v);
    # Km^TKm
    KTK = transpose(Km[:,1:m])*Km[:,1:m]
    # Km^T A Km
    KTAK = transpose(Km[:,1:m])*A*Km[:,1:m]
    # Solve generalized EVP
    Fg = eigen(KTAK,KTK)
    # find largest EigVal
    λ[m,1] = sort(abs.(Fg.values),rev=true)[1]

    # arnoldi
    Q,Hb = arnoldi(A,b,m,3)
    # extract Hm
    H = Hb[1:m,:]
    # find Ritz pair
    Fa = eigen(H)
    # find largest Eigval
    λ[m,2] = sort(abs.(Fa.values),rev=true)[1]
    println(λ[m,:])
end
figure()
x = 1:mmax
plot(x,λ[:,1])
plot(x,λ[:,2])
