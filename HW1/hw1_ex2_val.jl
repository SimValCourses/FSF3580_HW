# Exercise 2 - HW1
using LinearAlgebra
using PyPlot

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
    e_pm = (transpose(x)*A*x) ./ (transpose(x)*x);
    PM_err_v[k] = abs(e_pm - e_ref);
end
println("Eigvec error for power iteration is ",norm(A*x-e_pm*x))

# Rayleigh Quotient iteration
x = x0./norm(x0)
RQ_err_v = zeros(nmax,1)
mu = (transpose(x)*A*x) ./ (transpose(x)*x);
for k = 1:nmax
    global x, e_rq
    e_rq = mu
    x = (A-e_rq*I)\x;
    x = x/norm(x);
    e_rq = (transpose(x)*A*x) ./ (transpose(x)*x);
    RQ_err_v[k] = abs(e_rq - e_ref);
end

println("Eigvec error for Rayleigh quotient iteration is ", norm(A*x-e_rq*x))

# Plot error
iter_v = 1:nmax
semilogy(iter_v, PM_err_v, color="blue", linestyle="-")
semilogy(iter_v, RQ_err_v, color="red", linestyle="-")
