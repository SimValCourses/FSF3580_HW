function naive_kpik_val(A,b,m)
    include("../HW1/arnoldi_val.jl")
    Q, H = arnoldi(A,b,m)
    G, H = arnoldi(inv(A),b,m)
