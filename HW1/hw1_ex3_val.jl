# Exercise 3 - HW1
using Pkg
Pkg.add("MatrixDepot")
Pkg.add("BenchmarkTools")
using MatrixDepot, Random, LinearAlgebra, BenchmarkTools
include("GS_val.jl")
include("arnoldi_val.jl")

nn=200;
Random.seed!(0)
A = matrixdepot("wathen",nn,nn)
b = rand(size(A,1))

kv = [5 10 20 50 100]
nrun = 1
orth    = zeros(length(kv),nrun)
time_vec = zeros(length(kv),nrun)
#x0, y0 = arnoldi(A,b,1)  # cold start
for i = 1:length(kv)
    global m = kv[i]
    display(@benchmark global Q, H = arnoldi($A,$b,$m) seconds=30)
    # println("1:st should be zero = ", norm(Q*H-A*Q[:,1:m]));
    # println("2:nd Should be zero = ", norm(Q'*Q-I));
    orth[i] = norm(Q'*Q - I)
end
display(orth)
