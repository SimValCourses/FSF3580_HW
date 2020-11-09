# Exercise 4 - HW1
using Pkg
Pkg.add("MatrixDepot")
Pkg.add("BenchmarkTools")
Pkg.add("PyPlot")
using MatrixDepot, Random, LinearAlgebra, BenchmarkTools, PyPlot
include("GS_val.jl")
include("arnoldi_val.jl")

nn=10;
Random.seed!(0)
A = matrixdepot("wathen",nn,nn)
b = rand(size(A,1))

kv = 1:5
for i = 1:length(kv)
    global m = kv[i]
    global K = zeros(size(A,1),m)
    for j = 1:m
        K[:,j] = (A^(j-1)*b)/norm(A^(j-1)*b)
    end
    V = eigen(K'*A*K,K'*K)
    global Q, H = arnoldi(A,b,m)
    W = eigen(H[1:m,1:m])
    figure()
    plot(m*ones(m,1), real(V.values), "*")
    plot(m*ones(m,1), real(W.values), "o")
    pygui(true)
end
