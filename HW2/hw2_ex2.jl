using LinearAlgebra, Random, SparseArrays, PyPlot, LaTeXStrings, Arpack

include("GMRES.jl")
include("GS.jl")

n = 100;
m = 100;
Random.seed!(0);

# a)
# plotting
fig, ax = subplots(4,2);
x = 1:m;
#fig2, ax2 = subplots(4,1, sharex = true)

av = [1 5 10 100];

for (i,α) in enumerate(av)
	global fig, ax

	A = sprand(n,n,0.5);		# n x n sparse matrix of density 0.5
	A = A + α*sparse(I,n,n);
	A = A/norm(A,1);
	b = rand(n);

	xe = A\b;					# exact solution
	(evA,v) = eigs(A, nev=m-2);

    ax[i,2].vlines(0,-0.002,0.002,color="k")
	ax[i,2].hlines(0,-0.002,0.012,color="k")
	ax[i,2].scatter(real(evA),imag(evA),10,marker="x")
    ax[i,2].set_aspect("equal")
	#plt.gcf().gca().set_aspect("equal")
	ax[i,2].set_xlim(-0.002,0.012)
	ax[i,2].set_ylim(-0.002,0.002)
	ax[i,2].set_title("α = $α")

	# GMRES solution
	@time xstar,res,err = gmres_xe(A,b,m,xe,3)

	ax[i,1].semilogy(x,res,label="residual norm")
	ax[i,1].semilogy(x,err,label="error norm")
	ax[i,1].set_ylim(1e-15,10)
end

# b) simple convergence rates
c = [ 0.0054 0.0060 0.0089 ]
r = [ 0.0048 0.0041 0.0012 ]
for i in 2:4
	global c,r
	c1 = c[i-1]
	r1 = r[i-1]
	cr = r1/c1;
	ax[i,2].add_artist(plt.Circle((c1,0),r1,fill=false,color="k"))
	println("Simple Convergence rate: $cr")
	if i == 4
		ax[i,1].plot(x,cr.^(x.-1),linestyle="dashed",color="k",label="convergence bound (1 disk)")
		ax[i,1].legend()
	else
		ax[i,1].plot(x,cr.^(x.-1),linestyle="dashed",color="k")
	end
end
c = [ 0.0017 0.0029 0.0080 ]
c2 = 0.010
r = [ 0.0012 0.0010 0.0004 ]
r2 = 0.0003;
for i in 2:4
	global c,r,c2,r2
	local c1,r1
	c1 = c[i-1]
	r1 = r[i-1]
	cr = sqrt(r1*(r1 + abs(c1 - c2))/(c1*c2));
	ax[i,2].add_artist(plt.Circle((c1,0),r1,fill=false,color="r"))
	ax[i,2].add_artist(plt.Circle((c2,0),r2,fill=false,color="r"))
	println("Complex Convergence rate: $cr")
	if i == 4
		ax[i,1].plot(x,cr.^(x.-1),linestyle="dotted",color="r",label="convergence bound (2 disks)")
		ax[i,1].set_xlabel("Iteration count")
		ax[i,1].legend()
	else
		ax[i,1].plot(x,cr.^(x.-1),linestyle="dotted",color="r")
	end
end

#=
# c)
av = [ 1 100 ]
nalpha = length(av)
mv = [ 5 10 20 50 100 ]
msteps = length(mv)
nv = [ 100 200 500 1000 ]
nsize  = length(nv)

bstime = zeros(nsize)
bresv  = zeros(nsize)
grtime = zeros(msteps,nsize)
gresv  = zeros(msteps,nsize)

for (k,α) in enumerate(av)
	println("α = $α")
	for (i,n) in enumerate(nv)
		A = sprand(n,n,0.5);		# n x n sparse matrix of density 0.5
		A = A + α*sparse(I,n,n);
		A = A/norm(A,1);
		b = rand(n);

	    println("  n = $n")
		bstime[i] = @elapsed xe = A\b;					# exact solution
		bresv[i] = norm(A*xe - b)/norm(b)

		# GMRES solution
		for (j,m) in enumerate(mv)
	        println("    m = $m")
			grtime[j,i] = @elapsed xstar,res,err = gmres_xe(A,b,m,xe,3)
			gresv[j,i]  = res[m]
		end
	end
end
println(bstime)
println(bresv)

display(grtime)
display(gresv)
=#
