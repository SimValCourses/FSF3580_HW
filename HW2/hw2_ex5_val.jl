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
x_cgn, res_cgn, err_cgn, time_cgn = my_cg_val(As,c,m, x_exact);
x_gmres, res_gmres, err_gmres, time_gmres = my_gmres_val(A,b,m, x_exact, 1e-18);

pygui(true)
figure(1)
semilogy(1:m, res_gmres, color="black", linestyle="--",
        label=L"$\textnormal{GMRES}$")
semilogy(1:m, res_cgn, color="red", linestyle="-",
        label=L"$\textnormal{CGN}$")
xlabel(L"$\textnormal{Iteration}$")
ylabel(L"$\frac{||Ax-b||_2}{||b||_2}$")
legend()

figure(2)
semilogy(time_gmres, res_gmres,color="black", linestyle="--",
        label=L"$\textnormal{GMRES}$")
semilogy(time_cgn, res_cgn, color="red", linestyle="-",
        label=L"$\textnormal{CGN}$")
xlabel(L"$\textnormal{CPU-time}$")
ylabel(L"$\frac{||Ax-b||_2}{||b||_2}$")
legend()
