# Exercise 3 - HW2
using Pkg
Pkg.add("MatrixDepot")
Pkg.add("BenchmarkTools")
Pkg.add("PyPlot")
Pkg.add("Optim")
using MatrixDepot, Random, LinearAlgebra, BenchmarkTools, PyPlot
using SparseArrays, LaTeXStrings,IterativeSolvers
using Optim
include("GMRES_val.jl")
include("CG_val.jl")

A = [2 1 0 0 0 0 0 0
    1 2 -1 0 0 0 0 0
    0 -1 2 1 0 0 0 0
    0  0 1 2 1 0 0 0
    0  0 0 1 2 1 0 0
    0  0 0 0 1 2 1 0
    0  0 0 0 0 1 2 1
    0  0 0 0 0 0 1 2];

A2 = A*A;
b = [1 1 0 0 0 0 0 0]';
Ab = A*b;
A2b = A2*b;
A3 = A*A*A;
A3b = A3*b;

K = [b Ab A2b A3b];
B = [1  3  6 -29
     0 -1 -5  21
     0  0  1   0
     0  0  0  -1];
C = K*B;

# Minimization
x0 = [1 1 1 1];
f(z) = z'*C'*A*C*z-2 .*b'*C*z+b'*(A\b);
result = optimize(f, x0, BFGS());
z = result.minimizer;
x_fmin = C*z;
x_cg = my_cg_val(A,b,4)
x_exact = A\b;
display(norm(x_exact-x_cg))
