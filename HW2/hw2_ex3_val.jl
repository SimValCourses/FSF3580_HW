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
b = float([1; 1; 0; 0; 0; 0; 0; 0]);
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

x_exact = A\b;
sub_dim = 4;
x_cg, res_v, err_v = my_cg_val(A,b,sub_dim, x_exact);
x_gmres, res_v_gmres, err_v_gmres = my_gmres_val(A,b,sub_dim, x_exact, 1e-16);
println("Error to the exact solution (CG): ",norm(x_exact-x_cg))
println("Error to the exact solution (GMRES): ",norm(x_exact-x_gmres))

# Minimization on CG
x0 = float([1; 1; 1; 1]);
f1(z) = z'*C'*A*C*z-2*b'*C*z+b'*(A\b);
result_cg = optimize(f1, x0, f_tol = 1e-12);
z_cg = result_cg.minimizer;
x_fmin_cg = C*z_cg;
println("Error between minimization and CG: ",norm(x_fmin_cg-x_cg))

# Minimization on GMRES
x0 = float([1; 1; 1; 1]);
f2(z) = norm(A*C*z-b);
result_gmres = optimize(f2, x0, f_tol = 1e-12);
z_gmres = result_gmres.minimizer;
x_fmin_gmres = C*z_gmres;
println("Error between minimization and GMRES: ",norm(x_fmin_gmres-x_gmres))
