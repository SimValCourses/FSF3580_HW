# Exercise 2 - HW2
using Pkg
Pkg.add("MatrixDepot")
Pkg.add("BenchmarkTools")
Pkg.add("PyPlot")
using MatrixDepot, Random, LinearAlgebra, BenchmarkTools, PyPlot
using SparseArrays, LaTeXStrings,IterativeSolvers
include("GMRES_val.jl")

n=100;
Random.seed!(5);
b = rand(n,1);
b = b/norm(b);

alpha_v = [1 5 10 100];
for j = 1:length(alpha_v)
    alpha = alpha_v[j];
    A = sprand(n,n,0.5) + alpha .*sparse(I,n,n);
    A = A/norm(A,1);

    x_exact = A\b;

    m = 100;
    res_v = zeros(m,1);
    err_v = zeros(m,1);
    for i = 1:m
        x = my_gmres_val(A,b,i);
        res_v[i] = norm((A*x .- b));
        err_v[i] = norm((x .- x_exact));
    end
    pygui(true)
    figure(2*j-1)
    semilogy(1:m, res_v, color="blue", linestyle="-",
            label=L"$\textnormal{Residual norm}$")
    semilogy(1:m, err_v, color="red", linestyle="--",
            label=L"$\textnormal{Error norm}$")

    # Eigenvalues of A
    evA = eigen(Array(A));
    figure(2*j)
    scatter(real(evA.values),imag(evA.values))
end
