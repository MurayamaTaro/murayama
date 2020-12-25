module Support

using Bspline
using LinearAlgebra

export supp_Bs_double,supp_Bs_double_periodic,supp_Gauss_double,supp_Gauss_double_periodic,supp_Gauss_single,
        supp_Gauss_single_periodic,supp_basis_number,supp_basis_number_periodic

# 同方向の2つの基底関数が重なる範囲の番号を配列で返す
function supp_Bs_double(i::Int64,d::Int64,p::Array{Int64,1},N::Array{Int64,1})
    i-p[d]>1 ? l=i-p[d] : l=1
    i+p[d]<N[d] ? r=i+p[d] : r=N[d]
    range = Vector(l:r)
    return range
end


# 同方向の2つの基底関数が重なる範囲の番号を配列で返す(周期性あり)
function supp_Bs_double_periodic(i::Int64,d::Int64,p::Array{Int64,1},N::Array{Int64,1})
    i-p[d]>1 ? l=i-p[d] : l=1
    i+p[d]<N[d] ? r=i+p[d] : r=N[d]
    range = Vector(l:r)
    
    if i <= p[d]
        plus = Vector(N[d]-(p[d]-i):N[d])
        range = unique(vcat(range,plus))
    end
    if i >= N[d]-p[d]+1
        plus = Vector(1:i-N[d]+p[d])
        range = unique(vcat(range,plus))
    end
    sort!(range)
    return range
end


# 同方向の2つの基底関数が重なる範囲のガウス積分点の番号を配列で返す
function supp_Gauss_double(i::Int64,j::Int64,d::Int64,p::Array{Int64,1},G::Array{Int64,1},
        nip::Array{Int64,1})
    if i < j
        l = (j-p[d]-1)*nip[d]+1
        r = i*nip[d]
    else
        l = (i-p[d]-1)*nip[d]+1
        r = j*nip[d]
    end
    if l < 1
        l = 1
    end
    if r > G[d]
        r = G[d]
    end
    range = Vector(l:r)
    return range
end


# 同方向の2つの基底関数が重なる範囲のガウス積分点の番号を配列で返す（周期性あり）
function supp_Gauss_double_periodic(i::Int64,j::Int64,d::Int64,p::Array{Int64,1},G::Array{Int64,1},
        nip::Array{Int64,1})
    if i < j
        l = (j-p[d]-1)*nip[d]+1
        r = i*nip[d]
        if l < 1
            l = 1
        end
        if r > G[d]
            r = G[d]
        end
        range = Vector(l:r)
        if i <= p[d]
            l = G[d]-(p[d]-i+1)*nip[d]+1
            if j <= p[d]
                l = G[d]-(p[d]-j+1)*nip[d]+1
                r = G[d]
            else
                r = j*nip[d]
            end
            plus = Vector(l:r)
            range = vcat(range,plus)
        end
    else
        l = (i-p[d]-1)*nip[d]+1
        r = j*nip[d]
        if l < 1
            l = 1
        end
        if r > G[d]
            r = G[d]
        end
        range = Vector(l:r)
        if j <= p[d]
            l = G[d]-(p[d]-j+1)*nip[d]+1
            if i <= p[d]
                l = G[d]-(p[d]-i+1)*nip[d]+1
                r = G[d]
            else
                r = i*nip[d]
            end
            plus = Vector(l:r)
            range = vcat(range,plus)
        end
    end
    return range
end


# 台に含まれるガウス積分点の番号を配列で返す
function supp_Gauss_single(i::Int64,d::Int64,p::Array{Int64,1},G::Array{Int64,1},nip::Array{Int64,1})
    l = (i-p[d]-1)*nip[d]+1
    r = i*nip[d]
    if l<1
        l = 1
    end
    if r>G[d]
        r = G[d]
    end
    range = Vector(l:r)
    return range
end


# 台に含まれるガウス積分点の番号を配列で返す（周期性あり）
function supp_Gauss_single_periodic(i::Int64,d::Int64,p::Array{Int64,1},G::Array{Int64,1},
        nip::Array{Int64,1})
    l = (i-p[d]-1)*nip[d]+1
    r = i*nip[d]
    if l<1
        l = 1
    end
    if r>G[d]
        r = G[d]
    end
    range = Vector(l:r)
    if i<=p[d]
        l = G[d]-(p[d]-i+1)*nip[d]+1
        r = G[d]
        plus = Vector(l:r)
        range = vcat(range,plus)
    end
    return range
end


# 与えられた点(0~1)を含むB-spline基底関数の番号の範囲を配列で返す
function supp_basis_number(d::Int64,p::Array{Int64,1},k::Array{Array{Float64,1},1},N::Array{Int64,1},
        t::Float64)
    beg = 0
    en = 0
    for i in 1:N[d]
        if k[d][i]<=t<=k[d][i+1]
            beg = i-p[d]
            en = i
            if beg<1
                beg = 1
            end
            range = Vector(beg:en)
            return range
        end
    end
end


# 与えられた積分点，評価点を含むB-spline基底関数の番号の範囲を配列で返す（周期性あり）
function supp_basis_number_periodic(d::Int64,p::Array{Int64,1},k::Array{Array{Float64,1},1},
        N::Array{Int64,1},t::Float64)
    beg = 0
    en = 0
    for i in p[d]+1:N[d]+p[d]
        if k[d][i]<=t<=k[d][i+1]
            beg = i-p[d]
            en = i
            range = Vector(beg:en)
            if en>N[d]
                en = N[d]
                range = Vector(beg:en)
                
                beg_plus = 1
                en_plus = i-N[d]
                plus = Vector(beg_plus:en_plus)
                range = vcat(plus,range)
            end
            return range
        end
    end
end


end