# Exercise 14 - HW4
using Pkg
#Pkg.add("PyPlot")
Pkg.add("MatrixEquations")
#Pkg.add("LargeMatrixEquations")
using LinearAlgebra, PyPlot, LaTeXStrings, Random, SparseArrays
using MatrixEquations#, LargeMatrixEquations
include("spectral_abscissa.jl")

n = 100;
Random.seed!(0); # reset random seed
alphav = [1;2;3;4;5;10;13.725];
new_s = zeros(n,1);
b = randn(n)

# Solve Lypunov equation for various α
for (i,α) in enumerate(alphav)
    A = sprandn(n, n, 0.1);
    s = spectral_abscissa(A);
    A = A - sparse(I,size(A, 1),size(A,1)) * (α + s);
    new_s[i] = spectral_abscissa(A);
    X = lyap(Matrix(A),b*b');
    w = svd(X);
    ss = w.S;
    println("For α  = ", α, ", σ_6 = ", ss[6])
    aa = α;
    pygui(true)
    figure(1)
    h1 = semilogy(w.S, label="α = $aa", marker="x")
    ylabel(L"$σ$")
    xlabel(L"$k$")
end
legend()



# Solve Lyapunov via K-PIK
err = zeros(length(alphav),1);
for (i,α) in enumerate(alphav)
    A = sprandn(n, n, 0.1);
    s = spectral_abscissa(A);
    A = A - sparse(I,size(A, 1),size(A,1)) * (α + s);
    new_s[i] = spectral_abscissa(A);
    X = lyap(Matrix(A),b*b');
    X_k = kpik(A,b*b',E=1;LE=1,m=5,tol=1e-9,tolY=1e-12,infoV=true);
    err[i] = norm(X-X_k);
    println("Error for α = ", α, " is ", err[i])
end
