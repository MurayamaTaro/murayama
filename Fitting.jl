module Fitting

using Bspline
using LinearAlgebra
using Support

export fitting_1d,fitting_annulus,fitting_spherical_shell

# 一次元弾性体のフィッティング
function fitting_1d(dim::Int64,P::Array{Int64,1},k::Array{Array{Float64,1},1},
        N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},tG::Array{Array{Float64,1},1},
        wG::Array{Array{Float64,1},1},G::Array{Int64,1},BsG::Array{Array{Float64,2},1},
        R_outside::Float64,R_ratio::Float64)
    r1 = R_outside*R_ratio
    r2 = R_outside
    r(t1) = r1+(r2-r1)*t1
    x = [zeros(Float64,G[1]) for d in 1:dim]
    for iG1 in 1:G[1]
        x[1][iG1] = r(tG[1][iG1])
    end
    
    # 剛性行列
    M = zeros(Float64,Nall,Nall)
    for I1 in 1:N[1]
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        for J1 in supp_Bs1
            supp_Gauss1 = supp_Gauss_double(I1,J1,1,P,G,nip)
            if I1 < J1
                break
            end 
            c1 = sum(wG[1][iG1]*BsG[1][I1,iG1]*BsG[1][J1,iG1] for iG1 in supp_Gauss1)
            M[I1,J1] = c1
            M[J1,I1] = M[I1,J1]
        end
    end
    
    # 右辺ベクトル
    f = [zeros(Float64,Nall) for d in 1:dim]
    for I1 in 1:N[1]
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        for iG1 in supp_Gauss1
            c1 = wG[1][iG1]*BsG[1][I1,iG1]
            f[1][I1] += c1*x[1][iG1]
        end
    end
    
    # 制御点を求める
    a = [zeros(Float64,Nall) for d in 1:dim]
    for d in 1:dim
        a[d] = M\f[d]
    end
    
    return a
end
# 円環のフィッティング
function fitting_annulus(dim::Int64,p::Array{Int64,1},k::Array{Array{Float64,1},1},N::Array{Int64,1},
        Nall::Int64,nip::Array{Int64,1},tG::Array{Array{Float64,1},1},
        wG::Array{Array{Float64,1},1},G::Array{Int64,1},BsG::Array{Array{Float64,2},1},
        R_outside::Float64)
    r1 = R_outside/2.0
    r2 = R_outside
    r(t1) = r1+(r2-r1)*t1
    θ(t2) = 2.0*π*t2
    x = [zeros(Float64,G[1],G[2]) for d in 1:dim]
    for iG1 in 1:G[1]
        for iG2 in 1:G[2]
            x[1][iG1,iG2] = r(tG[1][iG1])*cos(θ(tG[2][iG2]))
            x[2][iG1,iG2] = r(tG[1][iG1])*sin(θ(tG[2][iG2]))
        end
    end
    
    # 剛性行列
    M = zeros(Float64,Nall,Nall)
    for I1 in 1:N[1]
        supp_Bs1 = supp_Bs_double(I1,1,p,N)
        for I2 in 1:N[2]
            supp_Bs2 = supp_Bs_double_periodic(I2,2,p,N)
            I = (I1-1)*N[2]+I2
            for J1 in supp_Bs1
                supp_Gauss1 = supp_Gauss_double(I1,J1,1,p,G,nip)
                for J2 in supp_Bs2
                    supp_Gauss2 = supp_Gauss_double_periodic(I2,J2,2,p,G,nip)
                    J = (J1-1)*N[2]+J2
                    if I < J
                        break
                    end
                    
                    c1 = 0.0
                    for iG1 in supp_Gauss1
                        c1 += wG[1][iG1]*BsG[1][I1,iG1]*BsG[1][J1,iG1]
                    end
                    c2 = 0.0
                    for iG2 in supp_Gauss2
                        c2 += wG[2][iG2]*BsG[2][I2,iG2]*BsG[2][J2,iG2]
                    end
                    
                    M[I,J] = c1*c2
                    M[J,I] = M[I,J]
                end
            end
        end
    end
    
    # 右辺ベクトル
    f = [zeros(Float64,Nall) for d in 1:dim]
    for I1 in 1:N[1]
        supp_Gauss1 = supp_Gauss_single(I1,1,p,G,nip)
        for I2 in 1:N[2]
            supp_Gauss2 = supp_Gauss_single_periodic(I2,2,p,G,nip)
            I = (I1-1)*N[2]+I2
            for iG1 in supp_Gauss1
                for iG2 in supp_Gauss2
                    f[1][I] += wG[1][iG1]*wG[2][iG2]*BsG[1][I1,iG1]*BsG[2][I2,iG2]*x[1][iG1,iG2]
                    f[2][I] += wG[1][iG1]*wG[2][iG2]*BsG[1][I1,iG1]*BsG[2][I2,iG2]*x[2][iG1,iG2]
                end
            end
        end
    end
    
    # 制御点を求める
    a = [zeros(Float64,Nall) for d in 1:dim]
    for d in 1:dim
        a[d] = M\f[d]
    end
    
    return a
end

# 球殻のフィッティング
function fitting_spherical_shell(dim::Int64,P::Array{Int64,1},k::Array{Array{Float64,1},1},
        N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},tG::Array{Array{Float64,1},1},
        wG::Array{Array{Float64,1},1},G::Array{Int64,1},BsG::Array{Array{Float64,2},1},
        R_outside::Float64,R_ratio::Float64)
    r1 = R_outside*R_ratio
    r2 = R_outside
    r(t1) = r1+(r2-r1)*t1
    θ(t2) = π*t2
    ϕ(t3) = 2*π*t3
    x = [zeros(Float64,G[1],G[2],G[3]) for d in 1:dim]
    for iG1 in 1:G[1]
        for iG2 in 1:G[2]
            for iG3 in 1:G[3]
                x[1][iG1,iG2,iG3] = r(tG[1][iG1])*sin(θ(tG[2][iG2]))*cos(ϕ(tG[3][iG3]))
                x[2][iG1,iG2,iG3] = r(tG[1][iG1])*sin(θ(tG[2][iG2]))*sin(ϕ(tG[3][iG3]))
                x[3][iG1,iG2,iG3] = r(tG[1][iG1])*cos(θ(tG[2][iG2]))
            end
        end
    end
    
    # 剛性行列
    M = zeros(Float64,Nall,Nall)
    for I1 in 1:N[1]
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        for I2 in 1:N[2]
            supp_Bs2 = supp_Bs_double(I2,2,P,N)
            for I3 in 1:N[3]
                supp_Bs3 = supp_Bs_double_periodic(I3,3,P,N)
                I = (I1-1)*N[2]*N[3]+(I2-1)*N[3]+I3
                for J1 in supp_Bs1
                    supp_Gauss1 = supp_Gauss_double(I1,J1,1,P,G,nip)
                    for J2 in supp_Bs2
                        supp_Gauss2 = supp_Gauss_double(I2,J2,2,P,G,nip)
                        for J3 in supp_Bs3
                            supp_Gauss3 = supp_Gauss_double_periodic(I3,J3,3,P,G,nip)
                            J = (J1-1)*N[2]*N[3]+(J2-1)*N[3]+J3
                            if I < J
                                break
                            end 
                            c1 = sum(wG[1][iG1]*BsG[1][I1,iG1]*BsG[1][J1,iG1] for iG1 in supp_Gauss1)
                            c2 = sum(wG[2][iG2]*BsG[2][I2,iG2]*BsG[2][J2,iG2] for iG2 in supp_Gauss2)
                            c3 = sum(wG[3][iG3]*BsG[3][I3,iG3]*BsG[3][J3,iG3] for iG3 in supp_Gauss3)
                            M[I,J] = c1*c2*c3
                            M[J,I] = M[I,J]
                        end
                    end
                end
            end
        end
    end
    
    # 右辺ベクトル
    f = [zeros(Float64,Nall) for d in 1:dim]
    for I1 in 1:N[1]
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        for I2 in 1:N[2]
            supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
            for I3 in 1:N[3]
                supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
                I = (I1-1)*N[2]*N[3]+(I2-1)*N[3]+I3
                for iG1 in supp_Gauss1
                    for iG2 in supp_Gauss2
                        for iG3 in supp_Gauss3
                            c1 = wG[1][iG1]*wG[2][iG2]*wG[3][iG3]*
                                BsG[1][I1,iG1]*BsG[2][I2,iG2]*BsG[3][I3,iG3]
                            f[1][I] += c1*x[1][iG1,iG2,iG3]
                            f[2][I] += c1*x[2][iG1,iG2,iG3]
                            f[3][I] += c1*x[3][iG1,iG2,iG3]
                        end
                    end
                end
            end
        end
    end
    
    # 制御点を求める
    a = [zeros(Float64,Nall) for d in 1:dim]
    for d in 1:dim
        a[d] = M\f[d]
    end
    
    return a
end


end