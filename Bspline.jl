module Bspline

using LinearAlgebra
using FastGaussQuadrature

export Bs,Ḃs,Ḃsn,Bs_value,pref!,href!,add_knot!,periodic_condition!,div_I_2d,div_I_3d,div_idx,make_Gauss,
        make_BsG!,make_BsE!

function Bs(i::Int64,p::Int64,k,t)::Float64
    if(p==0)
        return k[i]≤t<k[i+1]||(k[i]≠k[i+1]==k[end]==t)
    else
        return (((k[i+p]-k[i]≠0) ? Bs(i,p-1,k,t)*(t-k[i])/(k[i+p]-k[i]) : 0)
        +((k[i+p+1]-k[i+1]≠0) ? Bs(i+1,p-1,k,t)*(k[i+p+1]-t)/(k[i+p+1]-k[i+1]) : 0))
    end
end


function Ḃs(i::Int64,p::Int64,k,t)::Float64
    return p*(((k[i+p]-k[i]≠0) ? Bs(i,p-1,k,t)/(k[i+p]-k[i]) : 0)
    -((k[i+p+1]-k[i+1]≠0) ? Bs(i+1,p-1,k,t)/(k[i+p+1]-k[i+1]) : 0))
end


function Ḃsn(n::Int64,i::Int64,p::Int64,k,t)::Float64
    if(n==0)
        return Bs(i,p,k,t)
    elseif(n==1)
        return Ḃs(i,p,k,t)
    else
        return p*(((k[i+p]-k[i]≠0) ? Ḃsn(n-1,i,p-1,k,t)/(k[i+p]-k[i]) : 0)
        -((k[i+p+1]-k[i+1]≠0) ? Ḃsn(n-1,i+1,p-1,k,t)/(k[i+p+1]-k[i+1]) : 0))
    end
end


# 点列ts上のB-splineの値
function Bs_value(p::Int64,k::Array{Float64,1},derivative::Int64,ts::Array{Float64,1})
    N = length(k)-p-1
    v = zeros(Float64,N,length(ts))
    for i in 1:N
        for j in 1:length(ts)
            v[i,j] = Ḃsn(derivative,i,p,k,ts[j])
        end
    end
    return v
end


# B-spline関数のリファインメント
function pref!(Pi::Array{Int64,1},P::Array{Int64,1},k::Array{Array{Float64,1},1})
    for d in 1:length(P)
        l = k[d][1]
        r = k[d][end]
        n = P[d]-Pi[d]
        prepend!(k[d],[l for i in 1:n])
        append!(k[d],[r for i in 1:n])
    end
end


function href!(P::Array{Int64,1},k::Array{Array{Float64,1},1},N::Array{Int64,1})
    for d in 1:length(P)
        for i in 1:N[d]-P[d]-1
            insert!(k[d],P[d]+1+i,i/(N[d]-P[d]))
        end
    end
end


# ノット列の右端にノットを追加する
function add_knot!(dim::Int64,p::Array{Int64,1},k::Array{Array{Float64,1},1},N::Array{Int64,1},Nall::Int64,
        d₊::Int64,numk₊::Int64,border::Float64)
    for i in 1:p[d₊]+1
        pop!(k[d₊])
    end
    append!(k[d₊],[border+(1.0-border)*(i-1)/numk₊ for i in 1:numk₊])
    for i in 1:p[d₊]+1
        append!(k[d₊],1.0)
    end
    N[d₊] += numk₊
    Nall = prod([N[d] for d in 1:dim])
end


# 周期性条件
function periodic_condition!(p::Array{Int64,1},k::Array{Array{Float64,1},1},N::Array{Int64,1},
        periodic_dim::Array{Int64,1})
    for pd in periodic_dim
        span = k[pd][p[pd]+2]-k[pd][1]
        for i in 1:p[pd]
            k[pd][i] = -span*(p[pd]+1-i)
            k[pd][end+1-i] = k[pd][end-p[pd]] + span*(p[pd]+1-i)
        end
    end
end


# IをI1,I2に分ける
function div_I_2d(I::Int64,N::Array{Int64,1})
    I1 = div(I-1,N[2])+1
    I2 = (I-1)%N[2]+1
    return I1,I2
end


# IをI1,I2,I3に分ける
function div_I_3d(I::Int64,N::Array{Int64,1})
    I1 = div(div(I-1,N[3]),N[2])+1
    I2 = div(I-1,N[3])%N[2]+1
    I3 = (I-1)%N[3]+1
    return I1,I2,I3
end


# idxをI,iに分ける
function div_idx(dim::Int64,idx::Int64)
    I = div(idx-1,dim)+1
    i = (idx-1)%dim+1
    return I,i
end

# ガウス積分点列,重み列を作る(1ノットスパンにつきnip個積分点をとる)
function make_Gauss(p::Array{Int64,1},k::Array{Array{Float64,1},1},N::Array{Int64,1},
        nip::Array{Int64,1})
    dim = length(p)
    
    G = zeros(Int64,dim)
    for d in 1:dim
        G[d] = nip[d]*(N[d]-p[d])
    end
    
    tG = [zeros(Float64,G[d]) for d in 1:dim]
    wG = [zeros(Float64,G[d]) for d in 1:dim]
    
    for d in 1:dim
        for i in 1:N[d]-p[d]
            l = k[d][p[d]+i]
            r = k[d][p[d]+i+1]
        
            nodes,weights = gausslegendre(nip[d])
            nodes = nodes*(r-l)/2.0.+(l+r)/2.0
            weights = weights*(r-l)/2.0
        
            iG = (i-1)*nip[d]+1
            tG[d][iG:iG+nip[d]-1] .= nodes
            wG[d][iG:iG+nip[d]-1] .= weights
        end
    end
    return tG,wG,G
end

# Gauss積分点上のB-spline基底関数の値
function make_BsG!(dim::Int64,p::Array{Int64,1},k::Array{Array{Float64,1},1},N::Array{Int64,1},
        Nall::Int64,periodic_dim::Array{Int64,1},tG::Array{Array{Float64,N} ,1} where N,
        wG::Array{Array{Float64,N} ,1} where N,G::Array{Int64,1})
    BsG = [zeros(Float64,N[d],G[d]) for d in 1:dim]
    BsG_t = [zeros(Float64,N[d],G[d]) for d in 1:dim]
    BsG_tt = [zeros(Float64,N[d],G[d]) for d in 1:dim]
    BsG_ttt = [zeros(Float64,N[d],G[d]) for d in 1:dim]
    for d in 1:dim
        BsG[d] = Bs_value(p[d],k[d],0,tG[d])
        BsG_t[d] = Bs_value(p[d],k[d],1,tG[d])
        BsG_tt[d] = Bs_value(p[d],k[d],2,tG[d])
        BsG_ttt[d] = Bs_value(p[d],k[d],3,tG[d])
    end
    
    # 周期性のある方向ではB-spline基底関数を次数個連結させる
    for pd in periodic_dim
        N[pd] = N[pd]-p[pd]
        Nall = prod([N[d] for d in 1:dim])
        
        BsG_new = [zeros(Float64,N[d],G[d]) for d in 1:dim]
        BsG_t_new = [zeros(Float64,N[d],G[d]) for d in 1:dim]
        BsG_tt_new = [zeros(Float64,N[d],G[d]) for d in 1:dim]
        BsG_ttt_new = [zeros(Float64,N[d],G[d]) for d in 1:dim]
        for i in 1:p[pd]
            BsG_new[pd][i,:] .= BsG[pd][i,:]+BsG[pd][N[pd]+i,:]
            BsG_t_new[pd][i,:] .= BsG_t[pd][i,:]+BsG_t[pd][N[pd]+i,:]
            BsG_tt_new[pd][i,:] .= BsG_tt[pd][i,:]+BsG_tt[pd][N[pd]+i,:]
            BsG_ttt_new[pd][i,:] .= BsG_ttt[pd][i,:]+BsG_ttt[pd][N[pd]+i,:]
        end
        for i in p[pd]+1:N[pd]
            BsG_new[pd][i,:] .= BsG[pd][i,:]
            BsG_t_new[pd][i,:] .= BsG_t[pd][i,:]
            BsG_tt_new[pd][i,:] .= BsG_tt[pd][i,:]
            BsG_ttt_new[pd][i,:] .= BsG_ttt[pd][i,:]
        end
        BsG[pd] = copy(BsG_new[pd])
        BsG_t[pd] = copy(BsG_t_new[pd])
        BsG_tt[pd] = copy(BsG_tt_new[pd])
        BsG_ttt[pd] = copy(BsG_ttt_new[pd])
    end
    return BsG,BsG_t,BsG_tt,BsG_ttt,N,Nall
end

# 評価点列，評価点列上のBsの値を返す
function make_BsE!(dim::Int64,p::Array{Int64,1},k::Array{Array{Float64,1},1},N::Array{Int64,1},
        Nall::Int64,periodic_dim::Array{Int64,1},tEl::Array{Int64,1},flag_add_knot::Bool,
        d₊::Int64,numE₊::Int64,border::Float64)
    if dim==1
        tE = [collect(range(0,1,length=tEl[1])),[1],[1]]
    elseif dim==2
        tE = [collect(range(0,1,length=tEl[1])),collect(range(0,1,length=tEl[2])),[1]]
    elseif dim==3
        tE = [collect(range(0,1,length=tEl[1])),collect(range(0,1,length=tEl[2])),
                collect(range(0,1,length=tEl[3]))]
    end
    
    # 部分的にh-refinementしているとき
    if flag_add_knot
        pop!(tE[d₊])
        append!(tE[d₊],[border+(1.0-border)*(i-1)/numE₊ for i in 1:numE₊])
        append!(tE[d₊],1.0)
        tEl[d₊] += numE₊
    end
    
    # 値を代入
    BsE = [zeros(Float64,N[d],tEl[d]) for d in 1:dim]
    BsE_t = [zeros(Float64,N[d],tEl[d]) for d in 1:dim]
    BsE_tt = [zeros(Float64,N[d],tEl[d]) for d in 1:dim]
    BsE_ttt = [zeros(Float64,N[d],tEl[d]) for d in 1:dim]
    for d in 1:dim
        BsE[d] = Bs_value(p[d],k[d],0,tE[d])
        BsE_t[d] = Bs_value(p[d],k[d],1,tE[d])
        BsE_tt[d] = Bs_value(p[d],k[d],2,tE[d])
        BsE_ttt[d] = Bs_value(p[d],k[d],3,tE[d])
    end
    
    # 周期性のある方向では端を連結する
    for pd in periodic_dim
        BsE_new = [zeros(Float64,N[d],tEl[d]) for d in 1:dim]
        for i in 1:p[pd]
            BsE_new[pd][i,:] .= BsE[pd][i,:]+BsE[pd][N[pd]+i,:]
        end
        for i in p[pd]+1:N[pd]
            BsE_new[pd][i,:] .= BsE[pd][i,:]
        end
        BsE[pd] = copy(BsE_new[pd])
    end
    
    return tE,tEl,BsE,BsE_t,BsE_tt,BsE_ttt
end


end
