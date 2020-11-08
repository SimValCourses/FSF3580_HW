using MAT, PyPlot, Statistics, Random
include("arnoldi.jl")
include("GS.jl")
Random.seed!(0)
bw = MAT.matread("Bwedge.mat")

eigvals = bw["B_eigvals"];
A = bw["B"];
n = length(eigvals)
modeigs = vec([ abs(eig+15) for eig in eigvals ])
real = vec([ eig.re for eig in eigvals ])
imag = vec([ eig.im for eig in eigvals ])

figure(1)
subplot(3,3,1)
scatter(real,imag)
# circumscribed circles
cv = [ -5+0.5im -25-15im -25+15im ]
rv = [ 14 27 27 ]
cl = ["black" "red" "blue"]
cr = zeros(3)
for i in 1:3
    scatter(real[i],imag[i],s=100,color=cl[i],marker="x")
    ctr = cv[i]
    rad = rv[i]
    scatter(ctr.re,ctr.im,color="black")
    plt.gcf().gca().add_artist(plt.Circle((ctr.re,ctr.im), rad, fill=false, color=cl[i]))
    cr[i] = rad/(abs(eigvals[i]-ctr))
    println("Convergence rate for EV $i ($(eigvals[i])): $(cr[i])")
end
plt.gcf().gca().set_aspect("equal")
ylim(-15,15)
xlim(-60,10)

# Arnoldi
b = randn(n)

Q,H = arnoldi(A,b,40,3);

ksteps = [2 4 8 10 20 30 40];
error = zeros(length(ksteps))

for (j,k) in enumerate(ksteps)
    subplot(3,3,j+1)
    # all eigvals
    scatter(real,imag)
    # Ritz estimates
    P = eigen(H[1:k,1:k])
    rl = [ eig.re for eig in P.values ]
    ig = [ eig.im for eig in P.values ]
    # error
    error[j] = abs.(eigvals[1]-P.values[1])

    scatter(rl,ig,s=50,marker="x",color="black")
    # eigvals to which we converge fastest
    for i in 1:3
        ev = eigvals[i]
        scatter(ev.re,ev.im,s=200,color="none",edgecolor="red")
    end
    plt.gcf().gca().set_aspect("equal")
    ylim(-15,15)
    xlim(-60,10)
    ylabel("Im")
    xlabel("Re")
    title("Ritz estimates after $k iterations")
end
figure(2)
semilogy(vec(ksteps),1e-10*ones(7,1),color="black")
xticks(vec(ksteps))
semilogy(vec(ksteps),error,label="|λ_1 - λ_exact|")
title("Convergence to the outermost eigenvalue")
legend()


# Shift and invert
σ = -11 + 2im
B = A - σ*I
figure(5)
Beigs = [ 1/(eig - σ) for eig in eigvals ]
Breal = [ eig.re for eig in Beigs ]
Bimag = [ eig.im for eig in Beigs ]
scatter(Breal,Bimag)
tidx = argmin(abs.(eigvals .- σ))
t = Beigs[tidx]
scatter(t.re,t.im,s=200,color="none",edgecolor="red")
ct = -0.7+0.55im
ρt = 1.8
plt.gcf().gca().add_artist(plt.Circle((ct.re,ct.im), ρt, fill=false, color="black"))
plt.gcf().gca().set_aspect("equal")
ylim(-3,3)
xlim(-3,5)
cr = ρt/(abs(Beigs[tidx]-ct))
println("Convergence rate for EV 1 ($(Beigs[tidx])): $cr")


b = randn(n)

Q,H = arnoldiSI(B,b,40,3);

for (j,k) in enumerate(ksteps)
    global tidx
    # Ritz estimates
    P = eigen(H[1:k,1:k])
    # Convert back
    eigsA = [ 1/eig + σ for eig in P.values ]
    # error
    error[j] = abs.(eigvals[tidx]-eigsA[k])
end
figure(6)
semilogy(vec(ksteps),1e-10*ones(7,1),color="black")
xticks(vec(ksteps))
semilogy(vec(ksteps),error,label="|λ_1 - λ_exact|")
title("Convergence to the outermost eigenvalue")
legend()

return
