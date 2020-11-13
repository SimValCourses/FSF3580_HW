using LinearAlgebra, Random, SparseArrays, PyPlot, LaTeXStrings, Arpack

include("GMRES.jl")
include("GS.jl")

n = 100;
m = 100;
Random.seed!(0);

# plotting
figure(1);
figure(2);
x = 1:m;

av = [1 5 10 100];

for (i,α) in enumerate(av)
	
	A = sprand(n,n,0.5);		# n x n sparse matrix of density 0.5
	A = A + α*sparse(I,n,n);
	A = A/norm(A,1);
	b = rand(n);
	
	xe = A\b;					# exact solution
	(evA,v) = eigs(A, nev=m);
	figure(2)
	scatter(real(evA),imag(evA),10,marker="x",label="α = $α")
	
	# GMRES solution
	@time xstar,res,err = gmres_xe(A,b,m,xe,3)
	
	figure(1)
	p2 = subplot(2,2,i)
	semilogy(x,res,label="residual norm")
	semilogy(x,err,label="error norm")
	title("α = $α")
end
legend()
figure(2)
title("Eigenvalues of A")
legend()
