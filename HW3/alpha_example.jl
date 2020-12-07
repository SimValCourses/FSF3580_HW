using LinearAlgebra, Random
function alpha_example(alpha,m)
# Usage:
#  A=alpha_example(alpha,m)
#
# Generates a matrix of size m (which is 20 unless
# it is specified). alpha is a parameter which should be
# in the interval (0,infty).
#
# The basic QR-method will behave in very different ways
# for large and small alpha
#
  if (~(alpha>0))
    error("alpha must be positive")
  end


  Random.seed!(0)
  a::Float64=2;
  d=(a.^(0:(m-1)))
  d[end-1]=d[end]*(0.99-1/(5*alpha))
  B=randn(m,m)
  A=B\(diagm(0=>d[:])*B)
  return A
end
