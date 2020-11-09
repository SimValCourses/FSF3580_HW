function CGS(Q,w,k)
    h = Q'*w
    y = w - Q*h
    β = norm(y)
    h = h[1:k]
    return h,β,y
end

function MGS(Q,w,k)
    y = copy(w)
    h = zeros(k,1)
    for i = 1:k
        h[i] = Q[:,i]'*y
        y = y - h[i]*Q[:,i]
    end
    β = norm(y)
    return h,β,y
end

function DCGS(Q,w,k)
    h = Q'*w
    y = w - Q*h
    g = Q'*y
    y = y - Q*g
    h = h + g
    β = norm(y)
    h = h[1:k]
    return h,β,y
end

function TCGS(Q,w,k)
    h = Q'*w
    y = w - Q*h
    g = Q'*y
    y = y - Q*g
    f = Q'*y
    y = y - Q*f
    h = h + g + f
    β = norm(y)
    h = h[1:k]
    return h,β,y
end
