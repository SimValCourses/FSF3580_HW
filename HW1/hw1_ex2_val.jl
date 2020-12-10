# Exercise 2 - HW1
using Pkg
using LinearAlgebra
#Pkg.add("PyPlot")
#Pkg.add("LaTeXStrings")
using PyPlot, LaTeXStrings
include("PM_val.jl")
include("RQI_val.jl")

# Define matrix
A = [1 2 3
     2 2 2
     3 2 9]

ee = eigvals(A) # compute exact eigenvalues
ee_sorted = sort(ee, by=abs, rev=true)
e_ref = ee_sorted[1]

x0 = [1; 1; 1]
nmax = 20
iter_v = 1:nmax

# Power iteration
x = x0./norm(x0);
x, e_pm, PM_err_v = PM(A, x, nmax, e_ref)
println("Eigvec error for power iteration is ",norm(A*x-e_pm*x))
cPM = ee_sorted[2]/ee_sorted[1];
thPM = (cPM.^(2*(iter_v)))./cPM;

# Rayleigh Quotient iteration
x = x0./norm(x0);
x, e_rq, RQ_err_v = RQI(A, x, nmax, e_ref)
println("Eigvec error for Rayleigh quotient iteration is ", norm(A*x-e_rq*x))
thRQ = zeros(nmax,1);
thRQ[1] = RQ_err_v[1];
for i = 2:nmax
    global thRQ[i] = RQ_err_v[i-1]^(3);
end

# Non-symmetric matrix
B = [1 2 4
     2 2 2
     3 2 9];

ee_B = eigvals(B) # compute exact eigenvalues
ee_sorted_B = sort(ee_B, by=abs, rev=true)
e_ref_B = ee_sorted_B[1]

y = x0./norm(x0)
y, e_rq_B, RQ_err_v_B = RQI(B, y, nmax, e_ref_B)

# Plotting
figure(1)
semilogy(iter_v, PM_err_v, color="blue", linestyle="-", label=L"$\textnormal{Power method}$")
xlabel(L"$\textnormal{Number of iterations}$")
ylabel(L"$\textnormal{Eigenvalue error}$")
xticks(0:20)
ylim(1e-14,1)
xlim(0,15)
semilogy(iter_v, thPM, color="blue", linestyle="--", label=L"$\textnormal{Theoretical power method}$")
pygui(true)
legend()

figure(2)
semilogy(iter_v, RQ_err_v, color="red", linestyle="-",
        label=L"$\textnormal{Rayleigh quotient}$")
semilogy(iter_v, thRQ, color="red", linestyle="--",
        label=L"$\textnormal{Theoretical Rayleigh quotient}$")
xticks(0:20)
xlabel(L"$\textnormal{Number of iterations}$")
ylabel(L"$\textnormal{Eigenvalue error}$")
ylim(1e-14,1)
xlim(0,15)
pygui(true)
legend()

figure(3)
semilogy(iter_v, RQ_err_v, color="red", linestyle="-",
        label=L"$\textnormal{Rayleigh quotient symmetric}$")
semilogy(iter_v, RQ_err_v_B, color="black", linestyle="-.",
        label=L"$\textnormal{Rayleigh quotient non-symmetric}$")
semilogy(iter_v, thRQ, color="red", linestyle="--",
        label=L"$\textnormal{Theoretical Rayleigh quotient}$")
xticks(0:20)
xlabel(L"$\textnormal{Number of iterations}$")
ylabel(L"$\textnormal{Eigenvalue error}$")
ylim(1e-14,1)
xlim(0,15)
pygui(true)
legend()
