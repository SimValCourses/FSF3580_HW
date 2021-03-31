using LinearAlgebra

include("hessenberg_red.jl")
include("alpha_example.jl")

for m in [10 100 200 300 400]
    A = alpha_example(1,m)
    println("m=$m")
    @time H=naive_hessenberg_red(A)
    @time H=improved_hessenberg_red(A)
    global H_n=naive_hessenberg_red(A)
    global H_i=improved_hessenberg_red(A)
end

# d)
function QRshift(A,σ)
    H = qr(A-σ*I);
    Hbar = H.R*H.Q + σ*I;
    return Hbar
end

for σ in [ 0 1 ]
    for ϵ in [10 1.0 0.4 0.1 1e-2 1e-4 1e-6 1e-8 1e-10]
        A = [[3 2];[ϵ 1]];
        Hbar = QRshift(A,σ)
        println("ϵ = $ϵ, σ = $σ:")
        println("\t|h_2,1| = $(abs(Hbar[2,1]))")
        println("\t$(abs(Hbar[2,1])/ϵ)")
    end
    println("\n")
end
