using MAT, PyPlot, Statistics
bw = MAT.matread("Bwedge.mat")

eigvals = bw["B_eigvals"];
neigs = length(eigvals)
modeigs = vec([ abs(eig+15) for eig in eigvals ])
real = vec([ eig.re for eig in eigvals ])
imag = vec([ eig.im for eig in eigvals ])

figure()
scatter(real,imag)
idx = sortperm(modeigs,rev=true)
b = idx[1:3]
scatter(real[b],imag[b],s=100,color="red",marker="x")
# circumscribed circles
cv = [ -5+0.5im -25-15im -25+15im ]
rv = [ 14 27 27 ]
cr = zeros(3)
for i in 1:3
    c = cv[i]
    ρ = rv[i]
    scatter(c.re,c.im,color="black")
    plt.gcf().gca().add_artist(plt.Circle((c.re,c.im), ρ, fill=false))
    cr[i] = ρ/(abs(eigvals[i]-c))
    println("Conovergence rate for EV $i ($(eigvals[i])): $(cr[i])")
end
plt.gcf().gca().set_aspect("equal")
