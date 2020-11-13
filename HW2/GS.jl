function GS(Q,w,k,v)
    if v == 1
        h,β,y = CGS(Q,w,k)
    elseif v == 2
        h,β,y = MGS(Q,w,k)
    elseif v == 3
        h,β,y = DCGS(Q,w,k)
    else v == 4
        h,β,y = TCGS(Q,w,k)
    end
    return h,β,y
end

function CGS(Q,w,k)
    h = Q'*w
    y = w - Q*h
    β = norm(y)
    h = h[1:k]
    return h,β,y
end

function MGS(Q,w,k)
    h = []
    y = w
    # for each colum
    for i in 1:k
        push!(h,transpose(Q[:,i])*w)
        # project out direction
        y = y - Q[:,i]*h[i]
    end
    # normalize
    β = norm(y)
    return h,β,y
end

function DCGS(Q,w,k)
    h,β,y = CGS(Q,w,k)
    g,β,y = CGS(Q,y,k)
    h + g
    β = norm(y)
    h = h[1:k]
    return h,β,y
end

function TCGS(Q,w,k)
    h,β,y = CGS(Q,w,k)
    g,β,y = CGS(Q,y,k)
    f,β,y = CGS(Q,y,k)
    h = h + g + f
    β = norm(y)
    h = h[1:k]
    return h,β,y
end
