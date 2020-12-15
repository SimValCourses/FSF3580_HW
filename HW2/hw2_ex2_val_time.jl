# Exercise 2 timing - HW2
using Pkg
#Pkg.add("MatrixDepot")
#Pkg.add("BenchmarkTools")
#Pkg.add("PyPlot")
#Pkg.add("LaTeXStrings")
#Pkg.add("IterativeSolvers")
using MatrixDepot, Random, LinearAlgebra, PyPlot, BenchmarkTools
using SparseArrays, LaTeXStrings, IterativeSolvers
include("GMRES_val.jl")

n=500;
Random.seed!(5);
b = rand(n,1);
b = b/norm(b);

alpha = 100;
println("Alpha = ", alpha, "  n = ", n);
A = sprand(n,n,0.5) + alpha .*sparse(I,n,n);
A = A/norm(A,1);
back = @timed x_exact = A\b;
t_exact = back.time;
println("Time for backslash = ", t_exact)
println("Res for backslash = ", norm(A*x_exact-b)/norm(b))

tol = 1e-25;
m = n;
println("Time for GMRES m = ", m)
x, res_v, err_v, time_v = my_gmres_val(A,b,m,x_exact, tol);
println("Resnorm = ", norm(A*x-b)/norm(b))

m_vec = [5 10 20 50 100];# 200];# 500];
for j = 1:length(m_vec)
    println("m = ", m_vec[j], "  Time is ", time_v[m_vec[j]],
            "  Resnorm = ", res_v[m_vec[j]])
end
