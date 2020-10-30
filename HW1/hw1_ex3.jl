include("GS.jl")
include("arnoldi.jl")
using MatrixDepot, Random, LinearAlgebra

nn=200;
Random.seed!(0)
A = matrixdepot("wathen",nn,nn)
b = rand(size(A)[1])

kv = [5 10 20 50 100]
tn = ["CGS" "MGS" "DCGS" "TCGS"]
timings = zeros(length(kv),length(tn))
orth    = zeros(length(kv),length(tn))
x0,y0 = arnoldi(A,b,1,1)  # cold start
for (i,v) in enumerate(tn)
    println("$v:")
    for (j,m) in enumerate(kv)
        local tout = @timed Q,H = arnoldi(A,b,m,i)
        # println("1:st should be zero = ", norm(Q*H-A*Q[:,1:m]));
        # println("2:nd Should be zero = ", norm(Q'*Q-I));
        timings[j,i] = tout.time
        println("\t $(timings[j,i])")
        orth[j,i] = norm(Q'*Q - I)
    end
end
display(orth)
display(timings)
