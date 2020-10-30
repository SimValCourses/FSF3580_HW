using PyPlot
include("PM.jl")
include("RQI.jl")

A = [1 2 3; 2 2 2; 3 2 9]
v0 = [1, 1, 1]
F = eigen(A)
seigs = sort(F.values,rev=true)
λ_exact = seigs[1]

# a) Power Method
cPM = (seigs[2]/seigs[1])^2
k_max = 15
PMerror = zeros(k_max,1)
v = v0
for k in 1:k_max
    global v
    v = PM(A,v,1)
    PMerror[k] = abs(RQ(A,v)-λ_exact)
end

# b) Rayleigh Quotient Iteration
cRQI = cPM^3
RQIerror = zeros(k_max,1)
v = v0
for k in 1:k_max
    global v
    v = RQI(A,v,1)
    RQIerror[k] = abs(RQ(A,v)-λ_exact)
end

figure()
x = 1:k_max
semilogy(x,PMerror, color="red",label="PM")
semilogy(x,RQIerror,color="blue",label="RQI")
thPM = cPM.^x
thRQI = cRQI.^x*1e4
semilogy(x,thPM, color="black",label="theo_PM")
semilogy(x,thRQI,color="black",linestyle="--",label="theo_RQI")
xlabel("Iteration")
ylabel("Error")
title("Convergence of the Power Method")
ylim(bottom=1e-16)
legend()
