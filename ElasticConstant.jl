module ElasticConstant

using LinearAlgebra

export delta,makeδ_Classical_constant,makeFSG_constant,make_base_list,makeSSG_constant

function delta(i::Int64,j::Int64)
    if i==j
        return 1
    else
        return 0
    end
end

# 古典弾性体の弾性定数 C,nonzero_C
function makeδ_Classical_constant(λ::Float64,μ::Float64,dim::Int64)
    if dim==2
        δ = [1.0 0.0
             0.0 1.0]
    elseif dim==3
        δ = [1.0 0.0 0.0
             0.0 1.0 0.0
             0.0 0.0 1.0]
    end
    C = zeros(Float64,dim,dim,dim,dim)
    list_nzero= zeros(Int64,dim^4)
    cnt=1
    for i in 1:dim
        for j in 1:dim
            for k in 1:dim
                for l in 1:dim
                    α = dim^3*(i-1) + dim^2*(j-1) + dim^1*(k-1) + l
                    C[i,j,k,l] = λ*δ[i,j]*δ[k,l]+μ*(δ[i,k]*δ[j,l]+δ[i,l]*δ[j,k])
                    if C[i,j,k,l] !=0.0
                        list_nzero[cnt] = α
                        cnt+=1
                    end
                end
            end
        end
    end
    list_nzero = list_nzero[list_nzero.>0]
    return δ,C,list_nzero
end

# FSGの弾性定数 A6,B,nonzero_A6,nonzero_B
function makeFSG_constant(a1::Float64,b2::Float64,dim::Int64)
    A6 = zeros(Float64,dim,dim,dim,dim,dim,dim)
    B = zeros(Float64,dim,dim,dim,dim,dim,dim)
    list_nzero = zeros(Int64,dim^6)
    
    cnt = 1
    for i in 1:dim
        for j in 1:dim
            for k in 1:dim
                for l in 1:dim
                    for m in 1:dim
                        for n in 1:dim
                            α = dim^5*(i-1) + dim^4*(j-1) + dim^3*(k-1) + 
                                dim^2*(l-1) + dim^1*(m-1) + n
                            D1 = zeros(Float64,5)
                            D1[1] = (delta(i,n)*delta(j,k)*delta(l,m)+delta(i,k)*delta(j,n)*
                                    delta(l,m)+delta(i,j)*delta(k,m)*delta(l,n)+delta(i,j)*
                                    delta(k,l)*delta(m,n))
                            D1[2] = delta(i,j)*delta(k,n)*delta(l,m)
                            D1[3] = (delta(i,m)*delta(j,k)*delta(l,n)+delta(i,k)*delta(j,m)*
                                    delta(l,n)+delta(i,l)*delta(j,k)*delta(m,n)+delta(i,k)*
                                    delta(j,l)*delta(m,n))
                            D1[4] = delta(i,m)*delta(j,l)*delta(k,n) + delta(i,l)*delta(j,m)*delta(k,n)
                            D1[5] = (delta(i,n)*delta(j,m)*delta(k,l)+delta(i,m)*delta(j,n)*
                                    delta(k,l)+delta(i,n)*delta(j,l)*delta(k,m)+delta(i,l)*
                                    delta(j,n)*delta(k,m))
                            for β in 1:5
                                A6[i,j,k,l,m,n] += a1/15.0*D1[β]
                                B[i,j,k,l,m,n] += b2/15.0*D1[β]
                            end
                            if A6[i,j,k,l,m,n] != 0.0
                                list_nzero[cnt] = α
                                cnt += 1
                            end
                        end
                    end
                end
            end
        end
    end
    list_nzero = list_nzero[list_nzero.>0]
    list_nzero_A6 = copy(list_nzero)
    list_nzero_B = copy(list_nzero)
    return A6,B,list_nzero_A6,list_nzero_B
end

# 添字の通し番号を分解する
function make_base_list(base_num::Int64,dim::Int64)
    baselist = [zeros(Int64,base_num) for i in 1:dim^base_num]
    if base_num==4
        for α in 1:dim^4
            p = div((α-1),dim^3)+1
            q = div((α-1)%dim^3,dim^2)+1
            r = div((α-1)%dim^3%dim^2,dim^1)+1
            s = div((α-1)%dim^3%dim^2%dim^1,dim^0)+1
            baselist[α] = [p,q,r,s]
        end
    end
    if base_num==6
        for α in 1:dim^6
            p = div((α-1),dim^5)+1
            q = div((α-1)%dim^5,dim^4)+1
            r = div((α-1)%dim^5%dim^4,dim^3)+1
            s = div((α-1)%dim^5%dim^4%dim^3,dim^2)+1
            t = div((α-1)%dim^5%dim^4%dim^3%dim^2,dim^1)+1
            u = div((α-1)%dim^5%dim^4%dim^3%dim^2%dim^1,dim^0)+1
            baselist[α] = [p,q,r,s,t,u]
        end
    end
    if base_num==8
        for α in 1:dim^8
            p = div(α-1,dim^7) + 1
            q = div((α-1)%dim^7,dim^6)+1
            r = div((α-1)%dim^7%dim^6,dim^5)+1
            s = div((α-1)%dim^7%dim^6%dim^5,dim^4)+1
            t = div((α-1)%dim^7%dim^6%dim^5%dim^4,dim^3)+1
            u = div((α-1)%dim^7%dim^6%dim^5%dim^4%dim^3,dim^2)+1
            v = div((α-1)%dim^7%dim^6%dim^5%dim^4%dim^3%dim^2,dim^1)+1
            w = div((α-1)%dim^7%dim^6%dim^5%dim^4%dim^3%dim^2%dim^1,dim^0)+1
            baselist[α] = [p,q,r,s,t,u,v,w]
        end
    end
    return baselist
end

function makeSSG_constant(b1::Float64,dim::Int64)
    if dim==2
    E = zeros(Float64,2,2,2,2,2,2,2,2)
    list_nzero = zeros(Int64,dim^8)
    cnt = 1
    for α in 1:2^8
        i = div(α-1,2^7) + 1
        j = div((α-1)%2^7,2^6)+1
        k = div((α-1)%2^7%2^6,2^5)+1
        l = div((α-1)%2^7%2^6%2^5,2^4)+1
        m = div((α-1)%2^7%2^6%2^5%2^4,2^3)+1
        n = div((α-1)%2^7%2^6%2^5%2^4%2^3,2^2)+1
        p = div((α-1)%2^7%2^6%2^5%2^4%2^3%2^2,2^1)+1
        q = div((α-1)%2^7%2^6%2^5%2^4%2^3%2^2%2^1,2^0)+1
        
        E[i,j,k,l,m,n,p,q] = (delta(i,j)*delta(k,l)*delta(m,n)*delta(p,q)
                +delta(i,j)*delta(k,l)*delta(m,p)*delta(n,q)
                +delta(i,j)*delta(k,l)*delta(m,q)*delta(n,p)
                +delta(i,j)*delta(k,m)*delta(l,n)*delta(p,q)
                +delta(i,j)*delta(k,m)*delta(l,p)*delta(n,q)
                +delta(i,j)*delta(k,m)*delta(l,q)*delta(n,p)
                +delta(i,j)*delta(k,n)*delta(l,m)*delta(p,q)
                +delta(i,j)*delta(k,n)*delta(l,p)*delta(m,q)
                +delta(i,j)*delta(k,n)*delta(l,q)*delta(m,p)
                +delta(i,j)*delta(k,p)*delta(l,m)*delta(n,q)
                +delta(i,j)*delta(k,p)*delta(l,n)*delta(m,q)
                +delta(i,j)*delta(k,p)*delta(l,q)*delta(m,n)
                +delta(i,j)*delta(k,q)*delta(l,m)*delta(n,p)
                +delta(i,j)*delta(k,q)*delta(l,n)*delta(m,p)
                +delta(i,j)*delta(k,q)*delta(l,p)*delta(m,n)
                +delta(i,k)*delta(j,l)*delta(m,n)*delta(p,q)
                +delta(i,k)*delta(j,l)*delta(m,p)*delta(n,q)
                +delta(i,k)*delta(j,l)*delta(m,q)*delta(n,p)
                +delta(i,k)*delta(j,m)*delta(l,n)*delta(p,q)
                +delta(i,k)*delta(j,m)*delta(l,p)*delta(n,q)
                +delta(i,k)*delta(j,m)*delta(l,q)*delta(n,p)
                +delta(i,k)*delta(j,n)*delta(l,m)*delta(p,q)
                +delta(i,k)*delta(j,n)*delta(l,p)*delta(m,q)
                +delta(i,k)*delta(j,n)*delta(l,q)*delta(m,p)
                +delta(i,k)*delta(j,p)*delta(l,m)*delta(n,q)
                +delta(i,k)*delta(j,p)*delta(l,n)*delta(m,q)
                +delta(i,k)*delta(j,p)*delta(l,q)*delta(m,n)
                +delta(i,k)*delta(j,q)*delta(l,m)*delta(n,p)
                +delta(i,k)*delta(j,q)*delta(l,n)*delta(m,p)
                +delta(i,k)*delta(j,q)*delta(l,p)*delta(m,n)
                +delta(i,l)*delta(j,k)*delta(m,n)*delta(p,q)
                +delta(i,l)*delta(j,k)*delta(m,p)*delta(n,q)
                +delta(i,l)*delta(j,k)*delta(m,q)*delta(n,p)
                +delta(i,l)*delta(j,m)*delta(k,n)*delta(p,q)
                +delta(i,l)*delta(j,m)*delta(k,p)*delta(n,q)
                +delta(i,l)*delta(j,m)*delta(k,q)*delta(n,p)
                +delta(i,l)*delta(j,n)*delta(k,m)*delta(p,q)
                +delta(i,l)*delta(j,n)*delta(k,p)*delta(m,q)
                +delta(i,l)*delta(j,n)*delta(k,q)*delta(m,p)
                +delta(i,l)*delta(j,p)*delta(k,m)*delta(n,q)
                +delta(i,l)*delta(j,p)*delta(k,n)*delta(m,q)
                +delta(i,l)*delta(j,p)*delta(k,q)*delta(m,n)
                +delta(i,l)*delta(j,q)*delta(k,m)*delta(n,p)
                +delta(i,l)*delta(j,q)*delta(k,n)*delta(m,p)
                +delta(i,l)*delta(j,q)*delta(k,p)*delta(m,n)
                +delta(i,m)*delta(j,k)*delta(l,n)*delta(p,q)
                +delta(i,m)*delta(j,k)*delta(l,p)*delta(n,q)
                +delta(i,m)*delta(j,k)*delta(l,q)*delta(n,p)
                +delta(i,m)*delta(j,l)*delta(k,n)*delta(p,q)
                +delta(i,m)*delta(j,l)*delta(k,p)*delta(n,q)
                +delta(i,m)*delta(j,l)*delta(k,q)*delta(n,p)
                +delta(i,m)*delta(j,n)*delta(k,l)*delta(p,q)
                +delta(i,m)*delta(j,n)*delta(k,p)*delta(l,q)
                +delta(i,m)*delta(j,n)*delta(k,q)*delta(l,p)
                +delta(i,m)*delta(j,p)*delta(k,l)*delta(n,q)
                +delta(i,m)*delta(j,p)*delta(k,n)*delta(l,q)
                +delta(i,m)*delta(j,p)*delta(k,q)*delta(l,n)
                +delta(i,m)*delta(j,q)*delta(k,l)*delta(n,p)
                +delta(i,m)*delta(j,q)*delta(k,n)*delta(l,p)
                +delta(i,m)*delta(j,q)*delta(k,p)*delta(l,n)
                +delta(i,n)*delta(j,k)*delta(l,m)*delta(p,q)
                +delta(i,n)*delta(j,k)*delta(l,p)*delta(m,q)
                +delta(i,n)*delta(j,k)*delta(l,q)*delta(m,p)
                +delta(i,n)*delta(j,l)*delta(k,m)*delta(p,q)
                +delta(i,n)*delta(j,l)*delta(k,p)*delta(m,q)
                +delta(i,n)*delta(j,l)*delta(k,q)*delta(m,p)
                +delta(i,n)*delta(j,m)*delta(k,l)*delta(p,q)
                +delta(i,n)*delta(j,m)*delta(k,p)*delta(l,q)
                +delta(i,n)*delta(j,m)*delta(k,q)*delta(l,p)
                +delta(i,n)*delta(j,p)*delta(k,l)*delta(m,q)
                +delta(i,n)*delta(j,p)*delta(k,m)*delta(l,q)
                +delta(i,n)*delta(j,p)*delta(k,q)*delta(l,m)
                +delta(i,n)*delta(j,q)*delta(k,l)*delta(m,p)
                +delta(i,n)*delta(j,q)*delta(k,m)*delta(l,p)
                +delta(i,n)*delta(j,q)*delta(k,p)*delta(l,m)
                +delta(i,p)*delta(j,k)*delta(l,m)*delta(n,q)
                +delta(i,p)*delta(j,k)*delta(l,n)*delta(m,q)
                +delta(i,p)*delta(j,k)*delta(l,q)*delta(m,n)
                +delta(i,p)*delta(j,l)*delta(k,m)*delta(n,q)
                +delta(i,p)*delta(j,l)*delta(k,n)*delta(m,q)
                +delta(i,p)*delta(j,l)*delta(k,q)*delta(m,n)
                +delta(i,p)*delta(j,m)*delta(k,l)*delta(n,q)
                +delta(i,p)*delta(j,m)*delta(k,n)*delta(l,q)
                +delta(i,p)*delta(j,m)*delta(k,q)*delta(l,n)
                +delta(i,p)*delta(j,n)*delta(k,l)*delta(m,q)
                +delta(i,p)*delta(j,n)*delta(k,m)*delta(l,q)
                +delta(i,p)*delta(j,n)*delta(k,q)*delta(l,m)
                +delta(i,p)*delta(j,q)*delta(k,l)*delta(m,n)
                +delta(i,p)*delta(j,q)*delta(k,m)*delta(l,n)
                +delta(i,p)*delta(j,q)*delta(k,n)*delta(l,m)
                +delta(i,q)*delta(j,k)*delta(l,m)*delta(n,p)
                +delta(i,q)*delta(j,k)*delta(l,n)*delta(m,p)
                +delta(i,q)*delta(j,k)*delta(l,p)*delta(m,n)
                +delta(i,q)*delta(j,l)*delta(k,m)*delta(n,p)
                +delta(i,q)*delta(j,l)*delta(k,n)*delta(m,p)
                +delta(i,q)*delta(j,l)*delta(k,p)*delta(m,n)
                +delta(i,q)*delta(j,m)*delta(k,l)*delta(n,p)
                +delta(i,q)*delta(j,m)*delta(k,n)*delta(l,p)
                +delta(i,q)*delta(j,m)*delta(k,p)*delta(l,n)
                +delta(i,q)*delta(j,n)*delta(k,l)*delta(m,p)
                +delta(i,q)*delta(j,n)*delta(k,m)*delta(l,p)
                +delta(i,q)*delta(j,n)*delta(k,p)*delta(l,m)
                +delta(i,q)*delta(j,p)*delta(k,l)*delta(m,n)
                +delta(i,q)*delta(j,p)*delta(k,m)*delta(l,n)
                +delta(i,q)*delta(j,p)*delta(k,n)*delta(l,m))
        E[i,j,k,l,m,n,p,q] = b1/105 * E[i,j,k,l,m,n,p,q]
        if E[i,j,k,l,m,n,p,q] !=0
            list_nzero[cnt] = α
            cnt +=1
        end
    end
     list_nzero = list_nzero[list_nzero.>0]
    return E,list_nzero
    end
    if dim==3
     E = zeros(Float64,dim,dim,dim,dim,dim,dim,dim,dim)
    list_nzero = zeros(Int64,dim^8)
    cnt = 1
    for α in 1:dim^8
        i = div(α-1,dim^7) + 1
        j = div((α-1)%dim^7,dim^6)+1
        k = div((α-1)%dim^7%dim^6,dim^5)+1
        l = div((α-1)%dim^7%dim^6%dim^5,dim^4)+1
        m = div((α-1)%dim^7%dim^6%dim^5%dim^4,dim^3)+1
        n = div((α-1)%dim^7%dim^6%dim^5%dim^4%dim^3,dim^2)+1
        p = div((α-1)%dim^7%dim^6%dim^5%dim^4%dim^3%dim^2,dim^1)+1
        q = div((α-1)%dim^7%dim^6%dim^5%dim^4%dim^3%dim^2%dim^1,dim^0)+1
        
        E[i,j,k,l,m,n,p,q] = (delta(i,j)*delta(k,l)*delta(m,n)*delta(p,q)
                +delta(i,j)*delta(k,l)*delta(m,p)*delta(n,q)
                +delta(i,j)*delta(k,l)*delta(m,q)*delta(n,p)
                +delta(i,j)*delta(k,m)*delta(l,n)*delta(p,q)
                +delta(i,j)*delta(k,m)*delta(l,p)*delta(n,q)
                +delta(i,j)*delta(k,m)*delta(l,q)*delta(n,p)
                +delta(i,j)*delta(k,n)*delta(l,m)*delta(p,q)
                +delta(i,j)*delta(k,n)*delta(l,p)*delta(m,q)
                +delta(i,j)*delta(k,n)*delta(l,q)*delta(m,p)
                +delta(i,j)*delta(k,p)*delta(l,m)*delta(n,q)
                +delta(i,j)*delta(k,p)*delta(l,n)*delta(m,q)
                +delta(i,j)*delta(k,p)*delta(l,q)*delta(m,n)
                +delta(i,j)*delta(k,q)*delta(l,m)*delta(n,p)
                +delta(i,j)*delta(k,q)*delta(l,n)*delta(m,p)
                +delta(i,j)*delta(k,q)*delta(l,p)*delta(m,n)
                +delta(i,k)*delta(j,l)*delta(m,n)*delta(p,q)
                +delta(i,k)*delta(j,l)*delta(m,p)*delta(n,q)
                +delta(i,k)*delta(j,l)*delta(m,q)*delta(n,p)
                +delta(i,k)*delta(j,m)*delta(l,n)*delta(p,q)
                +delta(i,k)*delta(j,m)*delta(l,p)*delta(n,q)
                +delta(i,k)*delta(j,m)*delta(l,q)*delta(n,p)
                +delta(i,k)*delta(j,n)*delta(l,m)*delta(p,q)
                +delta(i,k)*delta(j,n)*delta(l,p)*delta(m,q)
                +delta(i,k)*delta(j,n)*delta(l,q)*delta(m,p)
                +delta(i,k)*delta(j,p)*delta(l,m)*delta(n,q)
                +delta(i,k)*delta(j,p)*delta(l,n)*delta(m,q)
                +delta(i,k)*delta(j,p)*delta(l,q)*delta(m,n)
                +delta(i,k)*delta(j,q)*delta(l,m)*delta(n,p)
                +delta(i,k)*delta(j,q)*delta(l,n)*delta(m,p)
                +delta(i,k)*delta(j,q)*delta(l,p)*delta(m,n)
                +delta(i,l)*delta(j,k)*delta(m,n)*delta(p,q)
                +delta(i,l)*delta(j,k)*delta(m,p)*delta(n,q)
                +delta(i,l)*delta(j,k)*delta(m,q)*delta(n,p)
                +delta(i,l)*delta(j,m)*delta(k,n)*delta(p,q)
                +delta(i,l)*delta(j,m)*delta(k,p)*delta(n,q)
                +delta(i,l)*delta(j,m)*delta(k,q)*delta(n,p)
                +delta(i,l)*delta(j,n)*delta(k,m)*delta(p,q)
                +delta(i,l)*delta(j,n)*delta(k,p)*delta(m,q)
                +delta(i,l)*delta(j,n)*delta(k,q)*delta(m,p)
                +delta(i,l)*delta(j,p)*delta(k,m)*delta(n,q)
                +delta(i,l)*delta(j,p)*delta(k,n)*delta(m,q)
                +delta(i,l)*delta(j,p)*delta(k,q)*delta(m,n)
                +delta(i,l)*delta(j,q)*delta(k,m)*delta(n,p)
                +delta(i,l)*delta(j,q)*delta(k,n)*delta(m,p)
                +delta(i,l)*delta(j,q)*delta(k,p)*delta(m,n)
                +delta(i,m)*delta(j,k)*delta(l,n)*delta(p,q)
                +delta(i,m)*delta(j,k)*delta(l,p)*delta(n,q)
                +delta(i,m)*delta(j,k)*delta(l,q)*delta(n,p)
                +delta(i,m)*delta(j,l)*delta(k,n)*delta(p,q)
                +delta(i,m)*delta(j,l)*delta(k,p)*delta(n,q)
                +delta(i,m)*delta(j,l)*delta(k,q)*delta(n,p)
                +delta(i,m)*delta(j,n)*delta(k,l)*delta(p,q)
                +delta(i,m)*delta(j,n)*delta(k,p)*delta(l,q)
                +delta(i,m)*delta(j,n)*delta(k,q)*delta(l,p)
                +delta(i,m)*delta(j,p)*delta(k,l)*delta(n,q)
                +delta(i,m)*delta(j,p)*delta(k,n)*delta(l,q)
                +delta(i,m)*delta(j,p)*delta(k,q)*delta(l,n)
                +delta(i,m)*delta(j,q)*delta(k,l)*delta(n,p)
                +delta(i,m)*delta(j,q)*delta(k,n)*delta(l,p)
                +delta(i,m)*delta(j,q)*delta(k,p)*delta(l,n)
                +delta(i,n)*delta(j,k)*delta(l,m)*delta(p,q)
                +delta(i,n)*delta(j,k)*delta(l,p)*delta(m,q)
                +delta(i,n)*delta(j,k)*delta(l,q)*delta(m,p)
                +delta(i,n)*delta(j,l)*delta(k,m)*delta(p,q)
                +delta(i,n)*delta(j,l)*delta(k,p)*delta(m,q)
                +delta(i,n)*delta(j,l)*delta(k,q)*delta(m,p)
                +delta(i,n)*delta(j,m)*delta(k,l)*delta(p,q)
                +delta(i,n)*delta(j,m)*delta(k,p)*delta(l,q)
                +delta(i,n)*delta(j,m)*delta(k,q)*delta(l,p)
                +delta(i,n)*delta(j,p)*delta(k,l)*delta(m,q)
                +delta(i,n)*delta(j,p)*delta(k,m)*delta(l,q)
                +delta(i,n)*delta(j,p)*delta(k,q)*delta(l,m)
                +delta(i,n)*delta(j,q)*delta(k,l)*delta(m,p)
                +delta(i,n)*delta(j,q)*delta(k,m)*delta(l,p)
                +delta(i,n)*delta(j,q)*delta(k,p)*delta(l,m)
                +delta(i,p)*delta(j,k)*delta(l,m)*delta(n,q)
                +delta(i,p)*delta(j,k)*delta(l,n)*delta(m,q)
                +delta(i,p)*delta(j,k)*delta(l,q)*delta(m,n)
                +delta(i,p)*delta(j,l)*delta(k,m)*delta(n,q)
                +delta(i,p)*delta(j,l)*delta(k,n)*delta(m,q)
                +delta(i,p)*delta(j,l)*delta(k,q)*delta(m,n)
                +delta(i,p)*delta(j,m)*delta(k,l)*delta(n,q)
                +delta(i,p)*delta(j,m)*delta(k,n)*delta(l,q)
                +delta(i,p)*delta(j,m)*delta(k,q)*delta(l,n)
                +delta(i,p)*delta(j,n)*delta(k,l)*delta(m,q)
                +delta(i,p)*delta(j,n)*delta(k,m)*delta(l,q)
                +delta(i,p)*delta(j,n)*delta(k,q)*delta(l,m)
                +delta(i,p)*delta(j,q)*delta(k,l)*delta(m,n)
                +delta(i,p)*delta(j,q)*delta(k,m)*delta(l,n)
                +delta(i,p)*delta(j,q)*delta(k,n)*delta(l,m)
                +delta(i,q)*delta(j,k)*delta(l,m)*delta(n,p)
                +delta(i,q)*delta(j,k)*delta(l,n)*delta(m,p)
                +delta(i,q)*delta(j,k)*delta(l,p)*delta(m,n)
                +delta(i,q)*delta(j,l)*delta(k,m)*delta(n,p)
                +delta(i,q)*delta(j,l)*delta(k,n)*delta(m,p)
                +delta(i,q)*delta(j,l)*delta(k,p)*delta(m,n)
                +delta(i,q)*delta(j,m)*delta(k,l)*delta(n,p)
                +delta(i,q)*delta(j,m)*delta(k,n)*delta(l,p)
                +delta(i,q)*delta(j,m)*delta(k,p)*delta(l,n)
                +delta(i,q)*delta(j,n)*delta(k,l)*delta(m,p)
                +delta(i,q)*delta(j,n)*delta(k,m)*delta(l,p)
                +delta(i,q)*delta(j,n)*delta(k,p)*delta(l,m)
                +delta(i,q)*delta(j,p)*delta(k,l)*delta(m,n)
                +delta(i,q)*delta(j,p)*delta(k,m)*delta(l,n)
                +delta(i,q)*delta(j,p)*delta(k,n)*delta(l,m))
        E[i,j,k,l,m,n,p,q] = b1/105 * E[i,j,k,l,m,n,p,q]
        if E[i,j,k,l,m,n,p,q] !=0
            list_nzero[cnt] = α
            cnt +=1
        end
    end
     list_nzero = list_nzero[list_nzero.>0]
    return E,list_nzero
    end       
end

end

