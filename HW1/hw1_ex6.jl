using MAT, PyPlot, Statistics, Random, LinearAlgebra
include("arnoldi.jl")
include("GS.jl")
Random.seed!(0)
bw = MAT.matread("Bwedge.mat")

evA = bw["B_eigvals"];
A = bw["B"];
n = length(evA)

# a) and b) eigvals with fastest convergence and convergence rate estimates
cv = [ -5+0.5im -25-15im -25+15im ]
rv = [ 14 27 27 ]
cl = ["k" "r" "b"]

figure(1)
scatter(real(evA),imag(evA))
for i in 1:3
    # eigenvalues that will converge fastest
    plot = scatter(real(evA)[i],imag(evA)[i],s=100,color=cl[i],marker="x")
    # center of circles
    scatter(real(cv[i]),imag(cv[i]),color="k")
    # circles
    plt.gcf().gca().add_artist(plt.Circle((real(cv[i]),imag(cv[i])), rv[i], fill=false, color=cl[i]))
    # convergence rate estimate
    cr = rv[i]/(abs(evA[i]-cv[i]))
    println("Convergence rate for EV $i ($(evA[i])): $cr")
end
plt.gcf().gca().set_aspect("equal")
ylim(-25,25); xlim(-60,10)
ylabel("Im"); xlabel("Re")

# c) Arnoldi method and Ritz estimates
b = randn(n)                    # starting vector
Q,H = arnoldi(A,b,40,3);        # run Arnoldi method

ksteps = [2 4 8 10 20 30 40];   # plot steps
tol = 1e-10                     # convergence criterion
ϵ_1 = zeros(ksteps[end])

Peigs = complex(zeros(ksteps[end],ksteps[end]))
for k in 1:ksteps[end]
    # Ritz estimates as eigvals of Hessenberg matrix
    P = eigen(H[1:k,1:k])
    Peigs[k,1:k] = P.values
    # error for first eigenvalue
    ϵ_1[k] = abs.(evA[1]-Peigs[k,1])
end
# find convergence step
idx = findfirst(x -> x < tol, ϵ_1)

# plotting
figure(2)
for (j,k) in enumerate(ksteps)
    subplot(4,2,j)
    # all eigvals
    plot = scatter(real(evA),imag(evA))
    # plot all estimates
    scatter(real(Peigs[j,1:k]),imag(Peigs[j,1:k]),s=50,marker="x",color="k")
    # mark eigvals to which we converge fastest (furthest away from cluster)
    for i in 1:3
        scatter(evA[i].re,evA[i].im,s=200,color="none",edgecolor="red")
    end
    plt.gcf().gca().set_aspect("equal")
    ylim(-15,15); xlim(-60,10)
    xlabel("Re")
    mod(j,2) == 1 ? ylabel("Im") : nothing
    title("Iteration $k")
end

# plot convergence of outermost eigenvalue
figure(3)
steps = 1:ksteps[end]
p = semilogy(steps,ϵ_1,label="|ϵ_1|")
semilogy(steps,1e-10*ones(ksteps[end]),color="k",label="convergence threshold")
vlines(steps[idx],1e-15,100,colors="k",linestyles="dashed",label="k = $idx")
xticks(vec(ksteps))
ylabel("|ϵ| = | λ - λ_exact |"); xlabel("iteration count k")
legend()

# d) Shift and invert
shiftv = [ -10 -7+2im -9.8+1.5im ]
figure(5)
idx = argmin(abs.(evA .- (-9.8 + 2im)))
subplot(2,2,1)
scatter(real(evA),imag(evA))
scatter(evA[idx].re,evA[idx].im,s=200,color="none",edgecolor="red",label="target")
title("Eigenvalues of A")
ylabel("Im")
legend()

for (i,σ) in enumerate(shiftv)
    local B, evB
    # choose shift & build B matrix
    σ = shiftv[i]
    B = A - σ*I
    # plot eigvals of B
    subplot(2,2,i+1)
    evB = [ 1/(eig - σ) for eig in evA ]
    plot = scatter(real(evB),imag(evB))
    # find eigval closes to shift in spectrum of A
    tidx = argmin(abs.(evA .- σ))
    scatter(real(evB[tidx]),imag(evB[tidx]),s=200,color="none",edgecolor="r")
    #plt.gcf().gca().set_aspect("equal")
    ylim(-2,1); xlim(-1,1)
    mod(i,2)==0 ? ylabel("Im") : nothing
    i>1 ? xlabel("Re") : nothing
    title("σ = $σ")
end
#suptitle("Eigenvalues of A and B(σ)")

b = randn(n)                        # starting vector
msteps = [ 10 20 30 ]               # output steps
println("Target: $(evA[idx])")
for σ in shiftv
    local B, Q, H
    B = A - σ*I
    Q,H = arnoldiSI(B,b,msteps[end],3); # Arnoldi shift and invert
    println("σ = $σ")
    for (j,m) in enumerate(msteps)
        global idx
        # Ritz estimates
        P = eigen(H[1:m,1:m])
        # Convert back
        evPc = [ 1/eig + σ for eig in P.values ]
        idx2 = argmin(abs.(evPc .- (-9.8 + 2im)))
        # error
        ϵ_SI = abs.(evA[idx] - evPc[idx2])
        println("\t m = $m: ϵ = $(ϵ_SI)")
    end
end
