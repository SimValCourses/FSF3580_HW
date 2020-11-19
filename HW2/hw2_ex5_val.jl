# Exercise 5 - HW2
Pkg.add("MAT")
using MAT, PyPlot, Statistics, Random, LinearAlgebra
include("GMRES_val.jl")
include("CG_val.jl")

bw = MAT.matread("Bwedge2.mat")
A = bw["B"];
b = bw["b"];
As = A'*A;
c = A'*b;

x_exact = A\b;
m = 100;
x_cgn, res_cgn, err_cgn = my_cg_val(As,c,m, x_exact);
x_gmres, res_gmres, err_gmres = my_gmres_val(A,b,m, x_exact, 1e-18);

pygui(true)
figure(1)
semilogy(1:m, res_gmres, color="blue", linestyle="-",
        label=L"$\textnormal{GMRES}$")
semilogy(1:m, res_cgn, color="red", linestyle="--",
        label=L"$\textnormal{CGN}$")
