# Exercise 3 - HW1
using Pkg
Pkg.add("MatrixDepot")
Pkg.add("BenchmarkTools")
using MatrixDepot, Random, LinearAlgebra, BenchmarkTools
include("GS.jl")
include("arnoldi.jl")

nn=10;
Random.seed!(0)
A = matrixdepot("wathen",nn,nn)
b = rand(size(A,1))

kv = [5 10 20 50 100]
orth    = zeros(length(kv),1)
#x0, y0 = arnoldi(A,b,1)  # cold start
for i = 1:length(kv)
    global m = kv[i]
    @btime global Q,H = arnoldi(A,b,m)
    # println("1:st should be zero = ", norm(Q*H-A*Q[:,1:m]));
    # println("2:nd Should be zero = ", norm(Q'*Q-I));
    orth[i] = norm(Q'*Q - I)
end
display(orth)
