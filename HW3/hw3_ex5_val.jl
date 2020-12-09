using LinearAlgebra, PyPlot, LaTeXStrings
include("schur_parlett_val.jl")

struct time_v
    naive
    parl
end

function naive_power(A, N)
    B = A;
    for i = 1:N-1
        B = B*A;
    end
    return B
end

# Test for sine function
A_test = [1 4 4; 3 -1 3; -1 4 4];
f = z->sin(z)
F = schur_parlett_val(A_test,f);
err_sin = norm(sin(A_test)-F);

# Power of a function
A = rand(100,100);
A = A/norm(A);
N_vec = 10:10:500;
t = time_v(zeros(length(N_vec),1), zeros(length(N_vec),1));

for (i,N) in enumerate(N_vec)
    g = z->z^N
    parl = @timed F = schur_parlett_val(A, g);
    naive = @timed B = naive_power(A, N)
    t.naive[i] = naive.time;
    t.parl[i] = parl.time;
    println("Error for Schur-Parlett N = ", N, " is ", norm(A^N-F))
    println("Error for naive N = ", N, " is ", norm(A^N-B))
end

pygui(true)
figure(1)
semilogy(N_vec, t.naive, color="black", linestyle="--",
        label=L"$\textnormal{Naive}$")
semilogy(N_vec, t.parl, color="red", linestyle="-",
        label=L"$\textnormal{Schur-Parlett}$")
xlabel(L"N")
ylabel(L"$\textnormal{CPU-time}$")
legend()
