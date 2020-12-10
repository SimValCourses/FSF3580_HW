# Exercise 6 - HW1
Pkg.add("MAT")
using MAT, PyPlot, Statistics, Random, LinearAlgebra
include("arnoldi_val.jl")
include("shift_invert_arnoldi_val.jl")
include("GS_val.jl")
Random.seed!(0)
bw = MAT.matread("../HW1/Bwedge.mat")

evA = bw["B_eigvals"];
A = bw["B"];
n = length(evA)

figure(1)
scatter(real(evA),imag(evA))
pygui(true)

b = randn(n)
#m_vec = [2 4 8 10 20 30 40]
#for i = 1:length(m_vec)
#    m = m_vec[i]
#    Q, H = arnoldi(A,b,m)
#    W = eigen(H[1:m,1:m])
#    figure(i+1)
#    scatter(real(evA),imag(evA))
#    scatter(real(W.values), imag(W.values), marker="x",color="k")
#end

Q,H = arnoldi(A,b,40);        # run Arnoldi method

ksteps = [2 4 8 10 20 30 40];   # plot steps
tol = 1e-10                     # convergence criterion
ϵ_1 = zeros(length(ksteps))

Peigs = complex(zeros(ksteps[end],ksteps[end]))
for (i,k) in enumerate(ksteps)
    # Ritz estimates as eigvals of Hessenberg matrix
    P = eigen(H[1:k,1:k])
    Peigs[i,1:k] = P.values
    # error for first eigenvalue
    ϵ_1[i] = min.(abs.(evA[1]-Peigs[i,1]))
end
for (j,k) in enumerate(ksteps)
    figure(j+1)
    scatter(real(evA),imag(evA))
    scatter(real(Peigs[j,1:k]),imag(Peigs[j,1:k]),s=50,marker="x",color="k")
end

# Shift-and-invert Arnoldi method
mu =-7+2im
steps = [10 20 30];
for i = 1:length(steps)
    n = steps[i]
    Q_si, H_si = shift_inv_arnoldi(A,b,mu,n)
    W_si = eigen(H_si[1:n,1:n])
    lambda = 1 ./W_si.values .+ mu

    # error
    idx = argmin(abs.(evA .- (-9.8 + 2im)))
    idx2 = argmin(abs.(lambda .- (-9.8 + 2im)))
    global ϵ_SI = abs.(evA[idx] - lambda[idx2])
    println("\t n = $n: ϵ = $(ϵ_SI)")

    figure(i+length(ksteps)+1)
    scatter(real(evA),imag(evA))
    scatter(real(lambda), imag(lambda), marker="x",color="k")
end
