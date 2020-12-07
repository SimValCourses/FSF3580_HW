using LinearAlgebra, Random, PyPlot, LaTeXStrings

include("schur_parlett_sk.jl")

# a)
#=
A = [[1 4 4];[3 -1 3];[-1 4 4]];
f = z->sin(z)
F = schur_parlett(A,f)
display(F)
=#

# b)
A = rand(100,100); A=A/norm(A);

Nmax = 500;
Nv = collect(1:25:Nmax);
timings = zeros(length(Nv),2)
G1 = naive_exp(A,10)
f = z->z^10
G2 = schur_parlett(A,f)
display(norm(G1-G2))
for (i,N) in enumerate(Nv)
    g=z->z^N
    tout = @timed B = naive_exp(A,N)
    timings[i,1] = tout.time
    tout = @timed B = schur_parlett(A,g)
    #@time T,Q=schur(Matrix{ComplexF64}(A));
    timings[i,2] = tout.time
    println("\t $(timings[i,:])")
end

nv = [5, 10, 50, 100, 200, 500]
N = 10;
timings2 = zeros(length(nv),2)
g=z->z^N
for (i,n) in enumerate(nv)
    T = rand(n,n); T=T/norm(T);
    G = naive_exp(A,10)
    G = schur_parlett(T,g)
    tout = @timed B = naive_exp(T,N)
    timings2[i,1] = tout.time
    tout = @timed B = schur_parlett(T,g)
    timings2[i,2] = tout.time
    println("\t $(timings[i,:])")
end

figure(1)
semilogy(Nv,timings[:,1],label="naive")
semilogy(Nv,timings[:,2],label="schur-parlett")
xlabel(L"N")
ylabel("CPU-time")
legend()

figure(2)
loglog(nv,timings2[:,1],label="naive")
loglog(nv,timings2[:,2],label="schur-parlett")
loglog(nv,(nv./5e3).^3,color="k",linestyle="dashed")
loglog(nv,(nv./5e3).^(2.5),color="r",linestyle="dashed")
loglog(nv,(nv./1e4).^2,color="b",linestyle="dashed")
xlabel(L"n")
ylabel("CPU-time")
legend()
