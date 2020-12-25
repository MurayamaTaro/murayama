module Elasticity3d

using Bspline
using LinearAlgebra
using Support
using Base.Threads

export calc_R_3d,calc_R_t_3d,R_tt,calc_J_Jinv_detJ_3d,calc_J_t_Jinv_t_3d,calc_wG_detJ_3d,
        calc_R_x_3d,R_xx,calc_x_t_3d,calc_normal_detJsrf_3d,calc_Q2_3d,calc_Q3_3d,
        calc_K1_3d,calc_K2_3d,calc_KP_ux_3d,calc_KP_uxx_3d,calc_F1_3d,calc_H1_3d,
        calc_F2_3d,calc_H2_3d,calc_F5_3d,calc_H5_3d,
        calc_rhs_surface_energy_3d,calc_rhs_hydrostatic_pressure_3d,calc_XE_uE_3d,calc_XG_uG_3d,calc_unG_3d


function calc_R_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1}
        ,G::Array{Int64,1},BsG::Array{Array{Float64,2},1})
    R = zeros(Float64,Nall,G[1],G[2],G[3])
     @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for iG1 in supp_Gauss1
            for iG2 in supp_Gauss2
                for iG3 in supp_Gauss3
                    R[I,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG[2][I2,iG2]*BsG[3][I3,iG3]
                end
            end
        end
    end
    return R
end


function calc_R_t_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1}
        ,G::Array{Int64,1},BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1})
    R_t = zeros(Float64,Nall,dim,G[1],G[2],G[3])
     @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for iG1 in supp_Gauss1
            for iG2 in supp_Gauss2
                for iG3 in supp_Gauss3
                    # i∈(1,2,3)の組み合わせ
                    R_t[I,1,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG[2][I2,iG2]*BsG[3][I3,iG3]
                    R_t[I,2,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG[3][I3,iG3]
                    R_t[I,3,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG[2][I2,iG2]*BsG_t[3][I3,iG3]
                end
            end
        end
    end
    return R_t
end


function R_tt(I::Int64,iG1::Int64,iG2::Int64,iG3::Int64,dim::Int64,N::Array{Int64,1},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},
        BsG_tt::Array{Array{Float64,2},1})
    I1,I2,I3 = div_I_3d(I,N)
    ans = zeros(Float64,dim,dim)
    ans[1,1] = BsG_tt[1][I1,iG1]*BsG[2][I2,iG2]*BsG[3][I3,iG3]
    ans[1,2] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG[3][I3,iG3]
    ans[1,3] = BsG_t[1][I1,iG1]*BsG[2][I2,iG2]*BsG_t[3][I3,iG3]
    ans[2,1] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG[3][I3,iG3]
    ans[2,2] = BsG[1][I1,iG1]*BsG_tt[2][I2,iG2]*BsG[3][I3,iG3]
    ans[2,3] = BsG[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
    ans[3,1] = BsG_t[1][I1,iG1]*BsG[2][I2,iG2]*BsG_t[3][I3,iG3]
    ans[3,2] = BsG[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
    ans[3,3] = BsG[1][I1,iG1]*BsG[2][I2,iG2]*BsG_tt[3][I3,iG3]
    return ans
end


function calc_J_Jinv_detJ_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},wG::Array{Array{Float64,1},1},
        a0::Array{Array{Float64,1},1},R_t::Array{Float64,5})
    J = zeros(Float64,dim,dim,G[1],G[2],G[3])
    Jinv = zeros(Float64,dim,dim,G[1],G[2],G[3])
    detJ = zeros(Float64,G[1],G[2],G[3])
    @threads for i in 1:dim
        for j in 1:dim
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
                                    J[i,j,iG1,iG2,iG3] += a0[i][I]*R_t[I,j,iG1,iG2,iG3]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    @threads for iG1 in 1:G[1]
        for iG2 in 1:G[2]
            for iG3 in 1:G[3]
                Jinv[:,:,iG1,iG2,iG3] = inv(J[:,:,iG1,iG2,iG3])
                detJ[iG1,iG2,iG3] = det(J[:,:,iG1,iG2,iG3])
            end
        end
    end
    return J,Jinv,detJ
end


function calc_J_t_Jinv_t_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},wG::Array{Array{Float64,1},1},
        a0::Array{Array{Float64,1},1},Jinv::Array{Float64,5},BsG::Array{Array{Float64,2},1},
        BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    J_t = zeros(Float64,dim,dim,dim,G[1],G[2],G[3])
    Jinv_t = zeros(Float64,dim,dim,dim,G[1],G[2],G[3])
    @threads for i in 1:dim
        for j in 1:dim
            for k in 1:dim
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
                                        _R_tt = R_tt(I,iG1,iG2,iG3,dim,N,BsG,BsG_t,BsG_tt)
                                        J_t[i,j,k,iG1,iG2,iG3] += a0[i][I]*_R_tt[j,k]
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    @threads for iG1 in 1:G[1]
        for iG2 in 1:G[2]
            for iG3 in 1:G[3]
                for i in 1:dim
                    for j in 1:dim
                        for k in 1:dim
                            c1 = 0.0
                            for m in 1:dim
                                for n in 1:dim
                                    c1 += (-Jinv[i,m,iG1,iG2,iG3]*J_t[m,n,k,iG1,iG2,iG3]*
                                                Jinv[n,j,iG1,iG2,iG3])
                                end
                            end
                            Jinv_t[i,j,k,iG1,iG2,iG3] = c1
                        end
                    end
                end
            end
        end
    end
    return J_t,Jinv_t
end


function calc_wG_detJ_3d(G::Array{Int64,1},wG::Array{Array{Float64,1},1},detJ::Array{Float64,3})
    wG_detJ = zeros(Float64,G[1],G[2],G[3])
    @threads for iG1 in 1:G[1]
        for iG2 in 1:G[2]
            for iG3 in 1:G[3]
                wG_detJ[iG1,iG2,iG3] = wG[1][iG1]*wG[2][iG2]*wG[3][iG3]*detJ[iG1,iG2,iG3]
            end
        end
    end
    return wG_detJ
end


function calc_R_x_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},wG::Array{Array{Float64,1},1},
        R_t::Array{Float64,5},Jinv::Array{Float64,5})
    R_x = zeros(Float64,Nall,dim,G[1],G[2],G[3])
     @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for i in 1:dim
            for iG1 in supp_Gauss1
                for iG2 in supp_Gauss2
                    for iG3 in supp_Gauss3
                        c1 = 0.0
                        for m in 1:dim
                            c1 += R_t[I,m,iG1,iG2,iG3]*Jinv[m,i,iG1,iG2,iG3]
                        end
                        R_x[I,i,iG1,iG2,iG3] = c1
                    end
                end
            end
        end
    end
    return R_x
end


function R_xx(I::Int64,i::Int64,j::Int64,iG1::Int64,iG2::Int64,iG3::Int64,dim::Int64,N::Array{Int64,1},
        R_t::Array{Float64,5},Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},BsG::Array{Array{Float64,2},1},
        BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    _R_tt = R_tt(I,iG1,iG2,iG3,dim,N,BsG,BsG_t,BsG_tt)
    c1 = 0.0
    c2 = 0.0
    for m in 1:dim
         for n in 1:dim
            c1 += _R_tt[m,n]*Jinv[m,i,iG1,iG2,iG3]*Jinv[n,j,iG1,iG2,iG3]
            c2 += R_t[I,m,iG1,iG2,iG3]*Jinv_t[m,i,n,iG1,iG2,iG3]*Jinv[n,j,iG1,iG2,iG3]
        end
    end
    return c1+c2
end

# function R_xx(I::Int64,i::Int64,j::Int64,iG1::Int64,iG2::Int64,iG3::Int64,dim::Int64,N::Array{Int64,1},
#         R_t::Array{Float64,5},Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},BsG::Array{Array{Float64,2},1},
#         BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
#     _R_tt = R_tt(I,iG1,iG2,iG3,dim,N,BsG,BsG_t,BsG_tt)
#     c1 = Jinv[:,:,iG1,iG2,iG3]'*_R_tt[:,:]*Jinv[:,:,iG1,iG2,iG3]
#     c2 = 0.0
#     for m in 1:dim
#          for n in 1:dim
            
#             c2 += R_t[I,m,iG1,iG2,iG3]*Jinv_t[m,i,n,iG1,iG2,iG3]*Jinv[n,j,iG1,iG2,iG3]
#         end
#     end
#     return c1+c2
# end


function calc_x_t_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},a::Array{Array{Float64,1},1},R_t::Array{Float64,5})
    x_t = zeros(Float64,dim,dim,G[2],G[3])
    for I1 in 1:N[1]
        for I2 in 1:N[2]
            supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
            for I3 in 1:N[3]
                supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
                I = (I1-1)*N[2]*N[3]+(I2-1)*N[3]+I3
                for iG2 in supp_Gauss2
                    for iG3 in supp_Gauss3
                        x_t[2,2,iG2,iG3] += a[2][I]*R_t[I,2,G[1],iG2,iG3]
                        x_t[3,3,iG2,iG3] += a[3][I]*R_t[I,3,G[1],iG2,iG3]
                        x_t[2,3,iG2,iG3] += a[2][I]*R_t[I,3,G[1],iG2,iG3]
                        x_t[3,2,iG2,iG3] += a[3][I]*R_t[I,2,G[1],iG2,iG3]
                        x_t[1,3,iG2,iG3] += a[1][I]*R_t[I,3,G[1],iG2,iG3]
                        x_t[1,2,iG2,iG3] += a[1][I]*R_t[I,2,G[1],iG2,iG3]
                    end
                end
            end
        end
    end
    return x_t
end


function calc_normal_detJsrf_3d(dim::Int64,G::Array{Int64,1},x_t::Array{Float64,4})
    normal = zeros(Float64,dim,G[2],G[3])
    @threads for iG2 in 1:G[2]
        for iG3 in 1:G[3]
            normal[1,iG2,iG3] = (x_t[2,2,iG2,iG3]*x_t[3,3,iG2,iG3]-
                                   x_t[2,3,iG2,iG3]*x_t[3,2,iG2,iG3])
            normal[2,iG2,iG3] = (x_t[1,3,iG2,iG3]*x_t[3,2,iG2,iG3]-
                                   x_t[1,2,iG2,iG3]*x_t[3,3,iG2,iG3])
            normal[3,iG2,iG3] = (x_t[1,2,iG2,iG3]*x_t[2,3,iG2,iG3]-
                                    x_t[1,3,iG2,iG3]*x_t[2,2,iG2,iG3])
        end
    end
    
    detJsrf = zeros(Float64,G[2],G[3])
    @threads for iG2 in 1:G[2]
        for iG3 in 1:G[3]
            norm = sqrt(normal[1,iG2,iG3]^2+normal[2,iG2,iG3]^2+normal[3,iG2,iG3]^2)
            detJsrf[iG2,iG3] = norm
            normal[1,iG2,iG3] /= norm
            normal[2,iG2,iG3] /= norm
            normal[3,iG2,iG3] /= norm
        end
    end
    return normal,detJsrf
end


function calc_Q2_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},a::Array{Array{Float64,1},1},R_x::Array{Float64,5})
    Q2 = zeros(Float64,dim,dim,G[1],G[2],G[3])
    @threads for i in 1:dim
        for j in 1:dim
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
                                    Q2[i,j,iG1,iG2,iG3] += a[i][I]*R_x[I,j,iG1,iG2,iG3]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return Q2
end


function calc_Q3_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},a::Array{Array{Float64,1},1},R_t::Array{Float64,5},
        Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    Q3 = zeros(Float64,dim,dim,dim,G[1],G[2],G[3])
    @threads for i in 1:dim
        for j in 1:dim
            for k in 1:dim
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
                                        Q3[i,j,k,iG1,iG2,iG3] += (a[i][I]*
                                                            R_xx(I,j,k,iG1,iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt))
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return Q3
end


function calc_K1_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},C::Array{Float64,4},R_x::Array{Float64,5})
    K1 = zeros(Float64,dim*Nall,dim*Nall)
    
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        supp_Bs2 = supp_Bs_double(I2,2,P,N)
        supp_Bs3 = supp_Bs_double_periodic(I3,3,P,N)
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
                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            for iG1 in supp_Gauss1
                                for iG2 in supp_Gauss2
                                    for iG3 in supp_Gauss3
                                        integrand = 0.0
                                        for j in 1:dim
                                            for l in 1:dim
                                                integrand += (C[i,j,k,l]*R_x[I,j,iG1,iG2,iG3]*
                                                            R_x[J,l,iG1,iG2,iG3])
                                            end
                                        end
                                        K1[idx1,idx2] += wG_detJ[iG1,iG2,iG3]*integrand
                                    end
                                end
                            end
                        end
                    end

                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            K1[idx2,idx1] = K1[idx1,idx2]
                        end
                    end
                end
            end
        end
    end
    return K1
end


function calc_K2_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},A6::Array{Float64,6},
        base4_list::Array{Array{Int64,1},1},R_t::Array{Float64,5},
        Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    K2 = zeros(Float64,dim*Nall,dim*Nall)
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        supp_Bs2 = supp_Bs_double(I2,2,P,N)
        supp_Bs3 = supp_Bs_double_periodic(I3,3,P,N)
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
                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            for iG1 in supp_Gauss1
                                for iG2 in supp_Gauss2
                                    for iG3 in supp_Gauss3
                                        integrand = 0.0
                                        for α in 1:dim^4
                                            j,l,m,n = base4_list[α]

                                            integrand += (A6[i,j,l,k,m,n]*
                                                        R_xx(I,j,l,iG1,iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt)*
                                                        R_xx(J,m,n,iG1,iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt))
                                        end
                                        K2[idx1,idx2] += wG_detJ[iG1,iG2,iG3]*integrand
                                    end
                                end
                            end
                        end
                    end

                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            K2[idx2,idx1] = K2[idx1,idx2]
                        end
                    end
                end
            end
        end
    end
    return K2
end


# 変位の法線方向の微分
function calc_KP_ux_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG::Array{Array{Float64,1},1},detJsrf::Array{Float64,2},γ::Float64,
        δ::Array{Float64,2},R_x::Array{Float64,5},normal::Array{Float64,3})
    KP = zeros(Float64,dim*Nall,dim*Nall)
    
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        supp_Bs2 = supp_Bs_double(I2,2,P,N)
        supp_Bs3 = supp_Bs_double_periodic(I3,3,P,N)
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
                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            for iG2 in supp_Gauss2
                                for iG3 in supp_Gauss3
                                    c1 = 0.0
                                    for p in 1:dim
                                        c1 += normal[p,iG2,iG3]*R_x[I,p,G[1],iG2,iG3]
                                    end
                                    c2 = 0.0
                                    for q in 1:dim
                                        c2 += normal[q,iG2,iG3]*R_x[J,q,G[1],iG2,iG3]
                                    end
                                    KP[idx1,idx2] += (wG[2][iG2]*wG[3][iG3]*detJsrf[iG2,iG3]*
                                                        2.0*γ*δ[i,k]*c1*c2)
                                end
                            end
                        end
                    end

                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            KP[idx2,idx1] = KP[idx1,idx2]
                        end
                    end
                end
            end
        end
    end
    return KP
end


# 変位の法線方向の2階微分
function calc_KP_uxx_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG::Array{Array{Float64,1},1},detJsrf::Array{Float64,2},γ::Float64,
        δ::Array{Float64,2},normal::Array{Float64,3},R_t::Array{Float64,5},
        Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    KP = zeros(Float64,dim*Nall,dim*Nall)
    
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        supp_Bs2 = supp_Bs_double(I2,2,P,N)
        supp_Bs3 = supp_Bs_double_periodic(I3,3,P,N)
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
                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            for iG2 in supp_Gauss2
                                for iG3 in supp_Gauss3
                                    c1 = 0.0
                                    for p in 1:dim
                                        for q in 1:dim
                                            c1 += (normal[p,iG2,iG3]*normal[q,iG2,iG3]*
                                                    R_xx(I,p,q,G[1],iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt))
                                        end
                                    end
                                    c2 = 0.0
                                    for p in 1:dim
                                        for q in 1:dim
                                            c2 += (normal[p,iG2,iG3]*normal[q,iG2,iG3]*
                                                    R_xx(J,p,q,G[1],iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt))
                                        end
                                    end
                                    KP[idx1,idx2] += (wG[2][iG2]*wG[3][iG3]*detJsrf[iG2,iG3]*
                                                        2.0*γ*δ[i,k]*c1*c2)
                                end
                            end
                        end
                    end

                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            KP[idx2,idx1] = KP[idx1,idx2]
                        end
                    end
                end
            end
        end
    end
    return KP
end


function calc_F1_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},δ::Array{Float64,2},C::Array{Float64,4},
        nonzero_C::Array{Int64,1},base4_list::Array{Array{Int64,1},1},R_x::Array{Float64,5},
        Q2::Array{Float64,5})
    F1 = zeros(Float64,dim*Nall)
    
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for i in 1:dim
            idx = (I-1)*dim+i

            for iG1 in supp_Gauss1
                for iG2 in supp_Gauss2
                    for iG3 in supp_Gauss3
                        integrand = 0.0
                        for α in nonzero_C
                            p,q,r,s = base4_list[α]

                            c1 = 0.0
                            for k in 1:dim
                                c1 += Q2[k,r,iG1,iG2,iG3]*Q2[k,s,iG1,iG2,iG3]
                            end
                            integrand += (1/2*C[p,q,r,s]*R_x[I,p,iG1,iG2,iG3]*
                                            Q2[i,q,iG1,iG2,iG3]*(c1-δ[r,s]))
                        end
                        F1[idx] += wG_detJ[iG1,iG2,iG3]*integrand
                    end
                end
            end

        end
    end
    
    return F1
end


function calc_H1_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},δ::Array{Float64,2},C::Array{Float64,4},
        nonzero_C::Array{Int64,1},base4_list::Array{Array{Int64,1},1},R_x::Array{Float64,5},
        Q2::Array{Float64,5})
    H1 = zeros(Float64,dim*Nall,dim*Nall)
   
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        supp_Bs2 = supp_Bs_double(I2,2,P,N)
        supp_Bs3 = supp_Bs_double_periodic(I3,3,P,N)
        for M1 in supp_Bs1
            supp_Gauss1 = supp_Gauss_double(I1,M1,1,P,G,nip)
            for M2 in supp_Bs2
                supp_Gauss2 = supp_Gauss_double(I2,M2,2,P,G,nip)
                for M3 in supp_Bs3
                    supp_Gauss3 = supp_Gauss_double_periodic(I3,M3,3,P,G,nip)
                    M = (M1-1)*N[2]*N[3]+(M2-1)*N[3]+M3
                    if I < M
                        break
                    end
                    for i in 1:dim
                        for m in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (M-1)*dim+m

                            for iG1 in supp_Gauss1
                                for iG2 in supp_Gauss2
                                    for iG3 in supp_Gauss3
                                        integrand = 0.0
                                        for α in nonzero_C
                                            p,q,r,s = base4_list[α]

                                            c1 = 0.0
                                            for k in 1:dim
                                                c1 += Q2[k,r,iG1,iG2,iG3]*Q2[k,s,iG1,iG2,iG3]
                                            end

                                            integrand += 1/2*C[p,q,r,s]*R_x[I,p,iG1,iG2,iG3]*
                                                    (δ[i,m]*R_x[M,q,iG1,iG2,iG3]*(c1-δ[r,s])+
                                                    2*R_x[M,r,iG1,iG2,iG3]*
                                                    Q2[i,q,iG1,iG2,iG3]*Q2[m,s,iG1,iG2,iG3])
                                        end
                                        H1[idx1,idx2] += wG_detJ[iG1,iG2,iG3]*integrand
                                    end
                                end 
                            end

                        end
                    end

                    for i in 1:dim
                        for m in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (M-1)*dim+m
                            H1[idx2,idx1] = H1[idx1,idx2]
                        end
                    end
                end
            end
        end
    end
    
    return H1
end


function calc_F2_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},A6::Array{Float64,6},nonzero_A6::Array{Int64,1},
        base6_list::Array{Array{Int64,1},1},R_x::Array{Float64,5},Q2::Array{Float64,5},Q3::Array{Float64,6},
        R_t::Array{Float64,5},Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    F2 = zeros(Float64,dim*Nall)
    
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for i in 1:dim
            idx = (I-1)*dim+i

            for iG1 in supp_Gauss1
                for iG2 in supp_Gauss2
                    for iG3 in supp_Gauss3
                        integrand = 0.0
                        for α in nonzero_A6
                            p,q,r,s,t,u = base6_list[α]

                            c1 = 0.0
                            for j in 1:dim
                                c1 += Q3[j,s,u,iG1,iG2,iG3]*Q2[j,t,iG1,iG2,iG3]
                            end

                            c2 = (Q2[i,q,iG1,iG2,iG3]*R_xx(I,p,r,iG1,iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt)+
                                    Q3[i,p,r,iG1,iG2,iG3]*R_x[I,q,iG1,iG2,iG3])

                            integrand += A6[p,q,r,s,t,u]*c1*c2
                        end
                        F2[idx] += wG_detJ[iG1,iG2,iG3]*integrand
                    end
                end
            end

        end
    end
    
    return F2
end


function calc_H2_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},δ::Array{Float64,2},A6::Array{Float64,6},
        nonzero_A6::Array{Int64,1},base6_list::Array{Array{Int64,1},1},
        R_x::Array{Float64,5},Q2::Array{Float64,5},Q3::Array{Float64,6},
        R_t::Array{Float64,5},Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    H2 = zeros(Float64,dim*Nall,dim*Nall)
    
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        supp_Bs2 = supp_Bs_double(I2,2,P,N)
        supp_Bs3 = supp_Bs_double_periodic(I3,3,P,N)
        for M1 in supp_Bs1
            supp_Gauss1 = supp_Gauss_double(I1,M1,1,P,G,nip)
            for M2 in supp_Bs2
                supp_Gauss2 = supp_Gauss_double(I2,M2,2,P,G,nip)
                for M3 in supp_Bs3
                    supp_Gauss3 = supp_Gauss_double_periodic(I3,M3,3,P,G,nip)
                    M = (M1-1)*N[2]*N[3]+(M2-1)*N[3]+M3
                    if I < M
                        break
                    end
                    for i in 1:dim
                        for m in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (M-1)*dim+m

                            for iG1 in supp_Gauss1
                                for iG2 in supp_Gauss2
                                    for iG3 in supp_Gauss3
                                        integrand = 0.0
                                        for α in nonzero_A6
                                            p,q,r,s,t,u = base6_list[α]

                                            c1 = 0.0
                                            for j in 1:dim
                                                c1 += Q3[j,s,u,iG1,iG2,iG3]*Q2[j,t,iG1,iG2,iG3]
                                            end
                                            
                                            c2 = (R_xx(I,p,r,iG1,iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt)*
                                                R_x[M,q,iG1,iG2,iG3] + 
                                                R_x[I,q,iG1,iG2,iG3]*
                                                R_xx(M,p,r,iG1,iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt))

                                            c3 = (R_xx(M,s,u,iG1,iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt)*
                                                Q2[m,t,iG1,iG2,iG3] + 
                                                R_x[M,t,iG1,iG2,iG3]*Q3[m,s,u,iG1,iG2,iG3])

                                            c4 = (Q2[i,q,iG1,iG2,iG3]*
                                                R_xx(I,p,r,iG1,iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt) + 
                                                Q3[i,p,r,iG1,iG2,iG3]*R_x[I,q,iG1,iG2,iG3])

                                            integrand += A6[p,q,r,s,t,u]*(δ[i,m]*c1*c2 + c3*c4)
                                        end
                                        H2[idx1,idx2] += wG_detJ[iG1,iG2,iG3]*integrand
                                    end
                                end
                            end

                        end
                    end

                    for i in 1:dim
                        for m in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (M-1)*dim+m
                            H2[idx2,idx1] = H2[idx1,idx2]
                        end
                    end
                    
                end
            end
        end
    end
    return H2
end


function calc_F5_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG::Array{Array{Float64,1},1},b0::Float64,detJsrf::Array{Float64,2},
        R_x::Array{Float64,5},Q2::Array{Float64,5},Q3::Array{Float64,6},
        normal::Array{Float64,3},R_t::Array{Float64,5},Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    F5 = zeros(Float64,dim*Nall)
    
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for i in 1:dim
            idx = (I-1)*dim+i

            for iG2 in supp_Gauss2
                for iG3 in supp_Gauss3
                    integrand = 0.0
                    for q in 1:dim
                        c1 = 0.0
                        c2 = 0.0
                        for p in 1:dim
                            c1 += R_xx(I,p,q,G[1],iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt)*Q2[i,p,G[1],iG2,iG3]
                            c2 += R_x[I,p,G[1],iG2,iG3]*Q3[i,p,q,G[1],iG2,iG3]
                        end
                        integrand += b0*normal[q,iG2,iG3]*(c1+c2)
                    end
                    F5[idx] += wG[2][iG2]*wG[3][iG3]*detJsrf[iG2,iG3]*integrand
                end
            end

        end
    end
    return F5
end


function calc_H5_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG::Array{Array{Float64,1},1},δ::Array{Float64,2},b0::Float64,
        detJsrf::Array{Float64,2},R_x::Array{Float64,5},normal::Array{Float64,3},R_t::Array{Float64,5},
        Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    H5 = zeros(Float64,dim*Nall,dim*Nall)
    
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Bs1 = supp_Bs_double(I1,1,P,N)
        supp_Bs2 = supp_Bs_double(I2,2,P,N)
        supp_Bs3 = supp_Bs_double_periodic(I3,3,P,N)
        for M1 in supp_Bs1
            for M2 in supp_Bs2
                supp_Gauss2 = supp_Gauss_double(I2,M2,2,P,G,nip)
                for M3 in supp_Bs3
                    supp_Gauss3 = supp_Gauss_double_periodic(I3,M3,3,P,G,nip)
                    M = (M1-1)*N[2]*N[3]+(M2-1)*N[3]+M3
                    if I < M
                        break
                    end
                    for i in 1:dim
                        for m in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (M-1)*dim+m

                            for iG2 in supp_Gauss2
                                for iG3 in supp_Gauss3
                                    integrand = 0.0
                                    for q in 1:dim
                                        c1 = 0.0
                                        c2 = 0.0
                                        for p in 1:dim
                                            c1 += R_xx(I,p,q,G[1],iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt)*
                                                R_x[M,p,G[1],iG2,iG3]
                                            c2 += R_x[I,p,G[1],iG2,iG3]*
                                                R_xx(M,p,q,G[1],iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt)
                                        end
                                        integrand += b0*δ[m,i]*normal[q,iG2,iG3]*(c1+c2)
                                    end
                                    H5[idx1,idx2] += (wG[2][iG2]*wG[3][iG3]*detJsrf[iG2,iG3]*
                                                        integrand)
                                end
                            end

                        end
                    end

                    for i in 1:dim
                        for m in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (M-1)*dim+m
                            H5[idx2,idx1] = H5[idx1,idx2]
                        end
                    end
                end
            end
        end
    end
    
    return H5
end



# 表面エネルギー項の右辺ベクトル
function calc_rhs_surface_energy_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},
        Nall::Int64,nip::Array{Int64,1},wG::Array{Array{Float64,1},1},
        G::Array{Int64,1},detJsrf::Array{Float64,2},b0::Float64,
        normal::Array{Float64,3},R_t::Array{Float64,5},
        Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},
        BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},BsG_tt::Array{Array{Float64,2},1})
    s = zeros(Float64,dim*Nall)
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for i in 1:dim
            idx = (I-1)*dim+i
            for iG2 in supp_Gauss2
                for iG3 in supp_Gauss3
                    c1 = 0.0
                    for j in 1:dim
                        c1 += normal[j,iG2,iG3]*R_xx(I,i,j,G[1],iG2,iG3,dim,N,R_t,Jinv,Jinv_t,BsG,BsG_t,BsG_tt)
                    end
                    s[idx] += wG[2][iG2]*wG[3][iG3]*detJsrf[iG2,iG3]*(-b0*c1)
                end
            end
        end
    end
    return s
end


# 静水圧の右辺ベクトル
function calc_rhs_hydrostatic_pressure_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},
        Nall::Int64,nip::Array{Int64,1},wG::Array{Array{Float64,1},1},
        G::Array{Int64,1},detJsrf::Array{Float64,2},normal::Array{Float64,3},BsG::Array{Array{Float64,2},1})
    s = zeros(Float64,dim*Nall)
    g = [0.0,-0.001,0.0]
    @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for i in 1:dim
            idx = (I-1)*dim+i
            for iG2 in supp_Gauss2
                for iG3 in supp_Gauss3
                    s[idx] += (wG[2][iG2]*wG[3][iG3]*detJsrf[iG2,iG3]*(-normal[i,iG2,iG3])*
                                BsG[2][I2,iG2]*BsG[3][I3,iG3])
                end
            end
        end
    end
    return s
end


# 評価点上の変位を求める
function calc_XE_uE_3d(dim::Int64,P::Array{Int64,1},k::Array{Array{Float64,1},1},
        N::Array{Int64,1},tE::Array{Array{Float64,1},1},tEl::Array{Int64,1},
        BsE::Array{Array{Float64,2},1},a0::Array{Array{Float64,1},1},uc::Array{Array{Float64,1},1})
    XE = zeros(Float64,3,tEl[1],tEl[2],tEl[3])
    uE = zeros(Float64,3,tEl[1],tEl[2],tEl[3])
    @threads for α in 1:tEl[1]*tEl[2]*tEl[3]
        iE1 = div(div(α-1,tEl[3]),tEl[2])+1
        iE2 = div(α-1,tEl[3])%tEl[2]+1
        iE3 = (α-1)%tEl[3]+1
        supp_basis_number1 = supp_basis_number(1,P,k,N,tE[1][iE1])
        supp_basis_number2 = supp_basis_number(2,P,k,N,tE[2][iE2])
        supp_basis_number3 = supp_basis_number_periodic(3,P,k,N,tE[3][iE3])
        for d in 1:dim
            for I1 in supp_basis_number1
                for I2 in supp_basis_number2
                    for I3 in supp_basis_number3
                        I = (I1-1)*N[2]*N[3]+(I2-1)*N[3]+I3
                        XE[d,iE1,iE2,iE3] += a0[d][I]*BsE[1][I1,iE1]*BsE[2][I2,iE2]*BsE[3][I3,iE3]
                        uE[d,iE1,iE2,iE3] += uc[d][I]*BsE[1][I1,iE1]*BsE[2][I2,iE2]*BsE[3][I3,iE3]
                    end
                end
            end
        end
    end
    return XE,uE
end


# 積分点上の変位を求める
function calc_XG_uG_3d(dim::Int64,P::Array{Int64,1},k::Array{Array{Float64,1},1},
        N::Array{Int64,1},G::Array{Int64,1},a0::Array{Array{Float64,1},1},uc::Array{Array{Float64,1},1},
        R::Array{Float64,4},R_x::Array{Float64,5},R_xx::Array{Float64,6},
        )
    XG = zeros(Float64,3,G[1],G[2],G[3])
    uG = zeros(Float64,3,G[1],G[2],G[3])
    u_xG = zeros(Float64,3,dim,G[1],G[2],G[3])
    u_xxG = zeros(Float64,3,dim,dim,G[1],G[2],G[3])

    @threads for α in 1:G[1]*G[2]*G[3]
        iG1 = div(div(α-1,G[3]),G[2])+1
        iG2 = div(α-1,G[3])%G[2]+1
        iG3 = (α-1)%G[3]+1
        supp_basis_number1 = supp_basis_number(1,P,k,N,tG[1][iG1])
        supp_basis_number2 = supp_basis_number(2,P,k,N,tG[2][iG2])
        supp_basis_number3 = supp_basis_number_periodic(3,P,k,N,tG[3][iG3])
        for I1 in supp_basis_number1
            for I2 in supp_basis_number2
                for I3 in supp_basis_number3
                    I = (I1-1)*N[2]*N[3]+(I2-1)*N[3]+I3
                    for i in 1:dim
                        XG[i,iG1,iG2,iG3] += a0[i][I]*R[I,iG1,iG2,iG3]
                        uG[i,iG1,iG2,iG3] += uc[i][I]*R[I,iG1,iG2,iG3]
                        for j in 1:dim
                            u_xG[i,j,iG1,iG2,iG3] += uc[i][I]*R_x[I,j,iG1,iG2,iG3]
                            for k in 1:dim
                                u_xxG[i,j,k,iG1,iG2,iG3] += uc[i][I]*R_xx[I,j,k,iG1,iG2,iG3]
                            end
                        end
                    end
                end
            end
        end
    end
    return XG,uG,u_xG,u_xxG
end


# 法線ベクトル方向の積分点上の変位を求める
function calc_unG_3d(dim::Int64,P::Array{Int64,1},k::Array{Array{Float64,1},1},
        N::Array{Int64,1},G::Array{Int64,1},a0::Array{Array{Float64,1},1},uc::Array{Array{Float64,1},1},
        R::Array{Float64,4},R_x::Array{Float64,5},R_xx::Array{Float64,6},normal::Array{Float64,3}
        )
    unG = zeros(Float64,G[1],G[2],G[3])

    @threads for α in 1:G[1]*G[2]*G[3]
        iG1 = div(div(α-1,G[3]),G[2])+1
        iG2 = div(α-1,G[3])%G[2]+1
        iG3 = (α-1)%G[3]+1
        supp_basis_number1 = supp_basis_number(1,P,k,N,tG[1][iG1])
        supp_basis_number2 = supp_basis_number(2,P,k,N,tG[2][iG2])
        supp_basis_number3 = supp_basis_number_periodic(3,P,k,N,tG[3][iG3])
        for I1 in supp_basis_number1
            for I2 in supp_basis_number2
                for I3 in supp_basis_number3
                    I = (I1-1)*N[2]*N[3]+(I2-1)*N[3]+I3
                    for i in 1:dim
                        unG[iG1,iG2,iG3] += uc[i][I]*R[I,iG1,iG2,iG3]*normal[i,iG2,iG3]
                    end
                end
            end
        end
    end
    return unG
end

end