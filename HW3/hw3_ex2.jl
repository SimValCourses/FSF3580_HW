using LinearAlgebra, PyPlot

include("alpha_example.jl")
include("myqr.jl")
#=
A = alpha_example(1e-2,10)
display(A)
F = eigen(A)
eigs = sort(abs.(F.values))
=#

tol = 1e-10

expv = collect(-1:1:5)
alphav = 10.0.^expv

n = 20;
itv = zeros(length(alphav))
stv = zeros(length(alphav))
for (i,α) in enumerate(alphav)
    global tol
    A = alpha_example(α,n);
    eigs = eigen(A)
    r = 1e-14;
    ip = 0;
    jp = 0;
    #eigs = sort(abs.(F.values))
    for (j,λ_1) in enumerate(eigs.values)
        for (k,λ_2) in enumerate(eigs.values[j+1:n])
            dn = max(abs(λ_1),abs(λ_2))
            up = min(abs(λ_1),abs(λ_2))
            cr = up/dn
            if cr > r
                r = cr;
                ip = j;
                jp = j+k;
            end
        end
    end
    stv[i] = log(tol)/log(r);
    println("Theoretical convergence rate: $r ($ip,$jp)\t$(stv[i]) steps")
    Am,QR,itv[i] = myqrtol(A,tol);
    println("Iteration $i: α = $α\t $(itv[i]) steps to convergence")
end

figure(1)
semilogx(alphav,itv)
semilogx(alphav,stv,color="k",linestyle="dashed")
