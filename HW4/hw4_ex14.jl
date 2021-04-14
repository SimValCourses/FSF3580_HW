# Exercise 14 - HW4
using Pkg
#Pkg.add("PyPlot")
#Pkg.add("MatrixEquations")
using LinearAlgebra, PyPlot, LaTeXStrings, Random, SparseArrays
using MatrixEquations
include("../HW1/arnoldi_val.jl")
include("../HW1/GS_val.jl")
include("spectral_abscissa.jl")
include("naive_kpik_val.jl")
include("kpik_val.jl")

n = 100;
Random.seed!(0); # reset random seed
alphav = [1;3;5;10;14.75;30];
new_s = zeros(n,1);
b = randn(n,1)

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
    kv = Vector(1:length(w.S))
    h1 = semilogy(kv, w.S, label="α = $α", marker="x")
    plot(kv, 1e-10*ones(length(kv),1), color="black", linestyle="--");
    ylabel(L"$\sigma$")
    xlabel(L"$k$")
end
PyPlot.yticks(10.0.^(-16:2:0))
legend()

# Solve Lyapunov via K-PIK
#err = zeros(length(alphav),1);
#for (i,α) in enumerate(alphav)
#    A = sprandn(n, n, 0.1);
#    s = spectral_abscissa(A);
#    A = A - sparse(I,size(A, 1),size(A,1)) * (α + s);
#    new_s[i] = spectral_abscissa(A);
#    X = lyap(Matrix(A),b*b');
#    E = 1; LE = 1;
#    mm = 5; tol = 1e-9; tolY = 1e-12; infoV = true;
#    X_k = kpik(Matrix(A),b,E,LE,mm, tol, tolY, infoV);
#    err[i] = norm(X-X_k);
#    println("Error for α = ", α, " is ", err[i])
#end

# Part b of the exercise
mvec = Vector(5:5:50);
err_appr = zeros(length(alphav),length(mvec));
for (i,α) in enumerate(alphav)
    A = sprandn(n, n, 0.1);
    s = spectral_abscissa(A);
    A = A - sparse(I,size(A, 1),size(A,1)) * (α + s);
    X = lyap(Matrix(A),b*b');
    for (j,m) in enumerate(mvec)
        Xt = naive_kpik_val(A,b,m)
        err_appr[i,j] = norm(X-Xt);
    end
end
for (i,α) in enumerate(alphav)
    figure(2)
    semilogy(mvec,err_appr[i,:],label="α = $α", marker = "x")
    plot(mvec, 1e-10*ones(length(mvec),1), color="black", linestyle="--")
    ylabel(L"$|| X ̃- X ||$")
    xlabel(L"$m$")
end
PyPlot.yticks(10.0.^(-16:2:0))
legend()

# Part c of the exercise
nvec = [50;100;200;500]

# Write teh vector of optimal m for each alpha (tol = 1e-10)
mopt = [39;19;14;10;9;7];
for (k,n) in enumerate(nvec)
    for (i,α) in enumerate(alphav)
        mm = mopt[i];
        println("α = ", α, "n = ", n)
        A = sprandn(n, n, 0.1);
        s = spectral_abscissa(A);
        A = A - sparse(I,size(A, 1),size(A,1)) * (α + s);
        b = randn(n,1)
        @time lyap(Matrix(A),b*b');
        @time naive_kpik_val(A,b,mm)
    end
end
