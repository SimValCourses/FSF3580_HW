# Exercise 2 - HW2
using Pkg
Pkg.add("MatrixDepot")
Pkg.add("BenchmarkTools")
Pkg.add("PyPlot")
Pkg.add("LaTeXStrings")
Pkg.add("IterativeSolvers")
using MatrixDepot, Random, LinearAlgebra, PyPlot, BenchmarkTools
using SparseArrays, LaTeXStrings, IterativeSolvers
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
    tol = 1e-20;
    x, res_v, err_v = my_gmres_val(A,b,m,x_exact, tol);
    pygui(true)
    figure(2*j-1)
    semilogy(1:m, res_v, color="blue", linestyle="-",
            label=L"$\textnormal{Residual norm}$")
    semilogy(1:m, err_v, color="red", linestyle="--",
            label=L"$\textnormal{Error norm}$")
    xlabel(L"$\textnormal{Number of iterations}$")
    ylabel(L"$\textnormal{Norm}$")
    legend()

    # Eigenvalues of A
    #B = Matrix(A);
    #evA = eigen(B);
    #figure(2*j)
    #scatter(real(evA.values),imag(evA.values))
    #xlabel(L"$\textnormal{Re}$")
    #ylabel(L"$\textnormal{Im}$")
end
