# Exercise 2 - HW1
using Pkg
using LinearAlgebra
#Pkg.add("PyPlot")
#Pkg.add("LaTeXStrings")
using PyPlot, LaTeXStrings


# Define matrix
A = [1 2 3
     2 2 2
     3 2 9]

ee = eigvals(A) # compute exact eigenvalues
ee_sorted = sort(ee, by=abs, rev=true)
e_ref = ee_sorted[1]

x0 = [1; 1; 1]
nmax = 20

# Power iteration
x = x0./norm(x0)
PM_err_v = zeros(nmax,1)
for k = 1:nmax
    global x, e_pm
    x = A*x;
    x = x/norm(x);
    e_pm = (x'*A*x) ./ (x'*x);
    PM_err_v[k] = abs(e_pm - e_ref);
end
println("Eigvec error for power iteration is ",norm(A*x-e_pm*x))
cPM = ee_sorted[2]/ee_sorted[1];

# Rayleigh Quotient iteration
x = x0./norm(x0)
RQ_err_v = zeros(nmax,1)
mu = (x'*A*x) ./ (x'*x);
for k = 1:nmax
    global x, e_rq
    e_rq = mu
    x = (A-e_rq*I)\x;
    x = x/norm(x);
    e_rq = (x'*A*x) ./ (x'*x);
    RQ_err_v[k] = abs(e_rq - e_ref);
end

println("Eigvec error for Rayleigh quotient iteration is ", norm(A*x-e_rq*x))

# Plot error
iter_v = 1:nmax
thPM = (cPM.^(2*(iter_v)))
figure()
semilogy(iter_v, PM_err_v, color="blue", linestyle="-", label=L"$\textnormal{Power method}$")
xlabel(L"$\textnormal{Number of iterations}$")
ylabel(L"$\textnormal{Eigenvalue error}$")
xticks(0:20)
ylim(1e-14,1)
xlim(0,15)
semilogy(iter_v, thPM, color="blue", linestyle="--", label=L"$\textnormal{Theoretical power method}$")
pygui(true)
legend()


thRQ = (RQ_err_v.^(3 .^(iter_v)))./(RQ_err_v[1]^2)
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

# Non-symmetric matrix
B = [1 2 4
     2 2 2
     3 2 9]

ee_B = eigvals(B) # compute exact eigenvalues
ee_sorted_B = sort(ee_B, by=abs, rev=true)
e_ref_B = ee_sorted_B[1]

y = x0./norm(x0)
RQ_err_v_B = zeros(nmax,1)
mu_B = (transpose(y)*B*y) ./ (transpose(y)*y);
for k = 1:nmax
    global y, e_rq_B
    e_rq_B = mu_B
    y = (B-e_rq_B*I)\y;
    y = y/norm(y);
    e_rq_B = (transpose(y)*B*y) ./ (transpose(y)*y);
    RQ_err_v_B[k] = abs(e_rq_B - e_ref_B);
end

figure()
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
