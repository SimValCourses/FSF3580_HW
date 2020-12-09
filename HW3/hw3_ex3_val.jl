using LinearAlgebra, PyPlot
include("QR_val.jl")
include("naive_hessenberg_red_val.jl")
include("hessenberg_red_val.jl")
include("alpha_example.jl")

struct time_v
    naive
    hessen
end

m_vec = [10; 100; 200; 300; 400];
time_m = zeros(length(m_vec), 2);
t = time_v(zeros(length(m_vec),1), zeros(length(m_vec),1));

for (i,m) in enumerate(m_vec)
    A = alpha_example(1, m)
    naive = @timed H_naive = naive_hessenberg_red_val(A);
    hess = @timed H = hessenberg_red_val(A);
    t.naive[i] = naive.time;
    t.hessen[i] = hess.time;
    println("m = ", m)
    println("Time for naive Hessenberg = ", naive.time)
    println("Time for Hessenberg reduction = ", hess.time)
end

expv = -1:-1:-10;
ϵ_vec = [0.4; 10.0.^expv];
σ_vec = [1; 10];
H_err = zeros(length(ϵ_vec), length(σ_vec));

for (i, ϵ) in enumerate(ϵ_vec)
    for (j, σ) in enumerate(σ_vec)
        A = [3 2; ϵ 1];
        H = shifted_QR_val(A, σ);
        H_err[i,j] = H[2,1];
        println("H_21 = ", H[2,1])
    end
end
