using LinearAlgebra

function RQ(A,v)
    return v'*A*v/norm(v)
end

function RQI(A,v,k)
    for i in 1:k
        v = (A - RQ(A,v)*I)\v
        v = v/norm(v)
    end
    return v
end
