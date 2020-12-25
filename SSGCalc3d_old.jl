module SSGCalc3d_old

using Bspline
using LinearAlgebra
using Support
using Base.Threads

export calc_R_3d,calc_R_t_3d,calc_R_tt_3d,calc_R_ttt_3d,calc_J_Jinv_detJ_3d,calc_J_t_Jinv_t_3d,calc_wG_detJ_3d,
        calc_R_x_3d,calc_R_xx_3d,calc_R_xxx_3d,calc_x_t_3d,calc_normal_detJsrf_3d,calc_Q2_3d,calc_Q3_3d,calc_Q4_3d,
        calc_K1_3d,calc_K2_3d,calc_K3_3d,calc_K4_3d,calc_KP_ux_3d,calc_KP_uxx_3d,calc_F1_3d,calc_H1_3d,
        calc_F2_3d,calc_H2_3d,calc_F3_3d,calc_H3_3d,calc_F4_3d,calc_H4_3d,calc_F5_3d,calc_H5_3d


# IをI1,I2,I3に分ける
function div_I_3d(I::Int64,N::Array{Int64,1})
    I1 = div(div(I-1,N[3]),N[2])+1
    I2 = div(I-1,N[3])%N[2]+1
    I3 = (I-1)%N[3]+1
    return I1,I2,I3
end


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


function calc_R_tt_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1}
        ,G::Array{Int64,1},BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},
        BsG_tt::Array{Array{Float64,2},1})
    R_tt = zeros(Float64,Nall,dim,dim,G[1],G[2],G[3])
     @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for iG1 in supp_Gauss1
            for iG2 in supp_Gauss2
                for iG3 in supp_Gauss3
                    # i,j∈(1,2,3)の組み合わせ
                    R_tt[I,1,1,iG1,iG2,iG3] = BsG_tt[1][I1,iG1]*BsG[2][I2,iG2]*BsG[3][I3,iG3]
                    R_tt[I,1,2,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG[3][I3,iG3]
                    R_tt[I,1,3,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_tt[I,2,1,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG[3][I3,iG3]
                    R_tt[I,2,2,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_tt[2][I2,iG2]*BsG[3][I3,iG3]
                    R_tt[I,2,3,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_tt[I,3,1,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_tt[I,3,2,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_tt[I,3,3,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG[2][I2,iG2]*BsG_tt[3][I3,iG3]
                end
            end
        end
    end
    return R_tt
end


function calc_R_ttt_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1}
        ,G::Array{Int64,1},BsG::Array{Array{Float64,2},1},BsG_t::Array{Array{Float64,2},1},
        BsG_tt::Array{Array{Float64,2},1},BsG_ttt::Array{Array{Float64,2},1})
    R_ttt = zeros(Float64,Nall,dim,dim,dim,G[1],G[2],G[3])
     @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for iG1 in supp_Gauss1
            for iG2 in supp_Gauss2
                for iG3 in supp_Gauss3
                    # i,j,k∈(1,2,3)の組み合わせ
                    R_ttt[I,1,1,1,iG1,iG2,iG3] = BsG_ttt[1][I1,iG1]*BsG[2][I2,iG2]*BsG[3][I3,iG3]
                    R_ttt[I,1,1,2,iG1,iG2,iG3] = BsG_tt[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG[3][I3,iG3]
                    R_ttt[I,1,1,3,iG1,iG2,iG3] = BsG_tt[1][I1,iG1]*BsG[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,1,2,1,iG1,iG2,iG3] = BsG_tt[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG[3][I3,iG3]
                    R_ttt[I,1,2,2,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_tt[2][I2,iG2]*BsG[3][I3,iG3]
                    R_ttt[I,1,2,3,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,1,3,1,iG1,iG2,iG3] = BsG_tt[1][I1,iG1]*BsG[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,1,3,2,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,1,3,3,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG[2][I2,iG2]*BsG_tt[3][I3,iG3]

                    R_ttt[I,2,1,1,iG1,iG2,iG3] = BsG_tt[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG[3][I3,iG3]
                    R_ttt[I,2,1,2,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_tt[2][I2,iG2]*BsG[3][I3,iG3]
                    R_ttt[I,2,1,3,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,2,2,1,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_tt[2][I2,iG2]*BsG[3][I3,iG3]
                    R_ttt[I,2,2,2,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_ttt[2][I2,iG2]*BsG[3][I3,iG3]
                    R_ttt[I,2,2,3,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_tt[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,2,3,1,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,2,3,2,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_tt[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,2,3,3,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_tt[3][I3,iG3]

                    R_ttt[I,3,1,1,iG1,iG2,iG3] = BsG_tt[1][I1,iG1]*BsG[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,3,1,2,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,3,1,3,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG[2][I2,iG2]*BsG_tt[3][I3,iG3]
                    R_ttt[I,3,2,1,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,3,2,2,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_tt[2][I2,iG2]*BsG_t[3][I3,iG3]
                    R_ttt[I,3,2,3,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_tt[3][I3,iG3]
                    R_ttt[I,3,3,1,iG1,iG2,iG3] = BsG_t[1][I1,iG1]*BsG[2][I2,iG2]*BsG_tt[3][I3,iG3]
                    R_ttt[I,3,3,2,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG_t[2][I2,iG2]*BsG_tt[3][I3,iG3]
                    R_ttt[I,3,3,3,iG1,iG2,iG3] = BsG[1][I1,iG1]*BsG[2][I2,iG2]*BsG_ttt[3][I3,iG3]
                end
            end
        end
    end
    return R_ttt
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
        a0::Array{Array{Float64,1},1},R_tt::Array{Float64,6},Jinv::Array{Float64,5})
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
                                        J_t[i,j,k,iG1,iG2,iG3] += a0[i][I]*R_tt[I,j,k,iG1,iG2,iG3]
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


function calc_J_tt_Jinv_tt_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},wG::Array{Array{Float64,1},1},
        a0::Array{Array{Float64,1},1},base4_list::Array{Array{Int64,1},1},
        R_tt::Array{Float64,6},R_ttt::Array{Float64,7},
        Jinv::Array{Float64,5},Jinv_t::Array{Float64,6})
    J_tt = zeros(Float64,dim,dim,dim,dim,G[1],G[2],G[3])
    Jinv_tt = zeros(Float64,dim,dim,dim,dim,G[1],G[2],G[3])
    @threads for α in 1:dim^4
        i,j,k,l = base4_list[α]
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
                                J_tt[i,j,k,l,iG1,iG2,iG3] += (a0[i][I]*
                                            R_ttt[I,j,k,l,iG1,iG2,iG3])
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
                            for l in 1:dim
                                c1 = 0.0
                                c2 = 0.0
                                c3 = 0.0
                                for m in 1:dim
                                    for n in 1:dim
                                        c1 += (-Jinv_t[i,m,l,iG1,iG2,iG3]*J_t[m,n,k,iG1,iG2,iG3]*
                                                Jinv[n,j,iG1,iG2,iG3])
                                        c2 += (-Jinv[i,m,iG1,iG2,iG3]*J_tt[m,n,k,l,iG1,iG2,iG3]*
                                                Jinv[n,j,iG1,iG2,iG3])
                                        c3 += (-Jinv[i,m,iG1,iG2,iG3]*J_t[m,n,k,iG1,iG2,iG3]*
                                                Jinv_t[n,j,l,iG1,iG2,iG3])
                                    end
                                end
                                Jinv_tt[i,j,k,l,iG1,iG2,iG3] = c1+c2+c3
                            end
                        end
                    end
                end
            end
        end
    end
    return J_tt,Jinv_tt
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


function calc_R_xx_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},wG::Array{Array{Float64,1},1},
        R_t::Array{Float64,5},R_tt::Array{Float64,6},Jinv::Array{Float64,5},Jinv_t::Array{Float64,6})
    R_xx = zeros(Float64,Nall,dim,dim,G[1],G[2],G[3])
     @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for i in 1:dim
            for j in 1:dim
                for iG1 in supp_Gauss1
                    for iG2 in supp_Gauss2
                        for iG3 in supp_Gauss3
                            c1 = 0.0
                            c2 = 0.0
                            for m in 1:dim
                                 for n in 1:dim
                                    c1 += (R_tt[I,m,n,iG1,iG2,iG3]*Jinv[m,i,iG1,iG2,iG3]*
                                            Jinv[n,j,iG1,iG2,iG3])
                                    c2 += (R_t[I,m,iG1,iG2,iG3]*Jinv_t[m,i,n,iG1,iG2,iG3]*
                                            Jinv[n,j,iG1,iG2,iG3])
                                end
                            end
                            R_xx[I,i,j,iG1,iG2,iG3] = c1+c2
                        end
                    end
                end
            end
        end
    end
    return R_xx
end


function calc_R_xxx_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},wG::Array{Array{Float64,1},1},
        R_t::Array{Float64,5},R_tt::Array{Float64,6},R_ttt::Array{Float64,7},
        Jinv::Array{Float64,5},Jinv_t::Array{Float64,6},Jinv_tt::Array{Float64,7})
    R_xxx = zeros(Float64,Nall,dim,dim,dim,G[1],G[2],G[3])
     @threads for I in 1:Nall
        I1,I2,I3 = div_I_3d(I,N)
        supp_Gauss1 = supp_Gauss_single(I1,1,P,G,nip)
        supp_Gauss2 = supp_Gauss_single(I2,2,P,G,nip)
        supp_Gauss3 = supp_Gauss_single_periodic(I3,3,P,G,nip)
        for i in 1:dim
            for j in 1:dim
                for k in 1:dim
                    for iG1 in supp_Gauss1
                        for iG2 in supp_Gauss2
                            for iG3 in supp_Gauss3
                                c1 = 0.0                                        
                                c2 = 0.0
                                c3 = 0.0
                                c4 = 0.0
                                c5 = 0.0
                                c6 = 0.0
                                for m in 1:dim
                                    for n in 1:dim
                                        for p in 1:dim
                                            c1 += (R_ttt[I,m,n,p,iG1,iG2,iG3]*
                                                    Jinv[m,i,iG1,iG2,iG3]*
                                                    Jinv[n,j,iG1,iG2,iG3]*
                                                    Jinv[p,k,iG1,iG2,iG3])
                                            c2 += (R_tt[I,m,n,iG1,iG2,iG3]*
                                                    Jinv_t[m,i,p,iG1,iG2,iG3]*
                                                    Jinv[n,j,iG1,iG2,iG3]*
                                                    Jinv[p,k,iG1,iG2,iG3])
                                            c3 += (R_tt[I,m,n,iG1,iG2,iG3]*
                                                    Jinv[m,i,iG1,iG2,iG3]*
                                                    Jinv_t[n,j,p,iG1,iG2,iG3]*
                                                    Jinv[p,k,iG1,iG2,iG3])
                                            c4 += (R_tt[I,m,p,iG1,iG2,iG3]*
                                                    Jinv_t[m,i,n,iG1,iG2,iG3]*
                                                    Jinv[n,j,iG1,iG2,iG3]*
                                                    Jinv[p,k,iG1,iG2,iG3])
                                            c5 += (R_t[I,m,iG1,iG2,iG3]*
                                                    Jinv_tt[m,i,n,p,iG1,iG2,iG3]*
                                                    Jinv[n,j,iG1,iG2,iG3]*
                                                    Jinv[p,k,iG1,iG2,iG3])
                                            c6 += (R_t[I,m,iG1,iG2,iG3]*
                                                    Jinv_t[m,i,n,iG1,iG2,iG3]*
                                                    Jinv_t[n,j,p,iG1,iG2,iG3]*
                                                    Jinv[p,k,iG1,iG2,iG3])
                                        end
                                    end
                                end
                                R_xxx[I,i,j,k,iG1,iG2,iG3] = c1+c2+c3+c4+c5+c6
                            end
                        end
                    end
                end
            end
        end
    end
    return R_xxx
end


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
        nip::Array{Int64,1},G::Array{Int64,1},a::Array{Array{Float64,1},1},R_xx::Array{Float64,6})
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
                                        Q3[i,j,k,iG1,iG2,iG3] += a[i][I]*R_xx[I,j,k,iG1,iG2,iG3]
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


function calc_Q4_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,
        nip::Array{Int64,1},G::Array{Int64,1},a::Array{Array{Float64,1},1},R_xxx::Array{Float64,7})
    Q4 = zeros(Float64,dim,dim,dim,dim,G[1],G[2],G[3])
    @threads for i in 1:dim
        for j in 1:dim
            for k in 1:dim
                for l in 1:dim
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
                                            Q4[i,j,k,l,iG1,iG2,iG3] += a[i][I]*R_xxx[I,j,k,l,iG1,iG2,iG3]
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return Q4
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
        base4_list::Array{Array{Int64,1},1},R_xx::Array{Float64,6})
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

                                            integrand += (A6[i,j,l,k,m,n]*R_xx[I,j,l,iG1,iG2,iG3]*
                                                            R_xx[J,m,n,iG1,iG2,iG3])
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


function calc_K3_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},A8::Array{Float64,8},
        base6_list::Array{Array{Int64,1},1},R_xxx::Array{Float64,7})
    K3 = zeros(Float64,dim*Nall,dim*Nall)
    
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
                                        for α in 1:dim^6
                                            j,l,m,n,p,q = base6_list[α]

                                            integrand += (A8[i,j,m,l,k,n,p,q]*
                                                            R_xxx[I,j,m,l,iG1,iG2,iG3]*
                                                            R_xxx[J,n,p,q,iG1,iG2,iG3])
                                        end
                                        K3[idx1,idx2] += wG_detJ[iG1,iG2,iG3]*integrand
                                    end
                                end
                            end
                        end
                    end

                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            K3[idx2,idx1] = K3[idx1,idx2]
                        end
                    end
                end
            end
        end
    end
    return K3
end


function calc_K4_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},B::Array{Float64,6},
        base4_list::Array{Array{Int64,1},1},R_x::Array{Float64,5},R_xxx::Array{Float64,7})
    K4 = zeros(Float64,dim*Nall,dim*Nall)
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

                                            c1 = (1/2*B[i,j,k,l,m,n]*R_xxx[I,l,m,n,iG1,iG2,iG3]*
                                                    R_x[J,j,iG1,iG2,iG3])
                                            c2 = (1/2*B[k,j,i,l,m,n]*R_x[I,j,iG1,iG2,iG3]*
                                                    R_xxx[J,l,m,n,iG1,iG2,iG3])

                                            integrand += c1+c2
                                        end
                                        K4[idx1,idx2] += wG_detJ[iG1,iG2,iG3]*integrand
                                    end
                                end
                            end
                        end
                    end

                    for i in 1:dim
                        for k in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (J-1)*dim+k
                            K4[idx2,idx1] = K4[idx1,idx2]
                        end
                    end
                end
            end
        end
    end
    return K4
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
        δ::Array{Float64,2},R_xx::Array{Float64,6},normal::Array{Float64,3})
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
                                                    R_xx[I,p,q,G[1],iG2,iG3])
                                        end
                                    end
                                    c2 = 0.0
                                    for p in 1:dim
                                        for q in 1:dim
                                            c2 += (normal[p,iG2,iG3]*normal[q,iG2,iG3]*
                                                    R_xx[J,p,q,G[1],iG2,iG3])
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
        base6_list::Array{Array{Int64,1},1},R_x::Array{Float64,5},R_xx::Array{Float64,6},
        Q2::Array{Float64,5},Q3::Array{Float64,6})
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

                            c2 = (Q2[i,q,iG1,iG2,iG3]*R_xx[I,p,r,iG1,iG2,iG3]+
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
        R_x::Array{Float64,5},R_xx::Array{Float64,6},Q2::Array{Float64,5},Q3::Array{Float64,6})
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
                                            
                                            c2 = (R_xx[I,p,r,iG1,iG2,iG3]*R_x[M,q,iG1,iG2,iG3] + 
                                                    R_x[I,q,iG1,iG2,iG3]*R_xx[M,p,r,iG1,iG2,iG3])

                                            c3 = (R_xx[M,s,u,iG1,iG2,iG3]*Q2[m,t,iG1,iG2,iG3] + 
                                                    R_x[M,t,iG1,iG2,iG3]*Q3[m,s,u,iG1,iG2,iG3])

                                            c4 = (Q2[i,q,iG1,iG2,iG3]*R_xx[I,p,r,iG1,iG2,iG3] + 
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


function calc_F3_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},A8::Array{Float64,8},
        nonzero_A8::Array{Int64,1},base8_list::Array{Array{Int64,1},1},
        R_x::Array{Float64,5},R_xx::Array{Float64,6},R_xxx::Array{Float64,7},Q2::Array{Float64,5},
        Q3::Array{Float64,6},Q4::Array{Float64,7})
    F3 = zeros(Float64,dim*Nall)
    
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
                        for α in nonzero_A8
                            p,q,r,s,t,u,v,w = base8_list[α]

                            c1_1 = 0.0
                            c1_2 = 0.0
                            for j in 1:dim
                                c1_1 += Q4[j,t,v,w,iG1,iG2,iG3]*Q2[j,u,iG1,iG2,iG3]
                                c1_2 += Q3[j,t,v,iG1,iG2,iG3]*Q3[j,u,w,iG1,iG2,iG3]
                            end
                            c1 = c1_1+c1_2

                            c2 = (Q4[i,p,r,s,iG1,iG2,iG3]*R_x[I,q,iG1,iG2,iG3]+
                                    2*Q3[i,q,s,iG1,iG2,iG3]*R_xx[I,p,r,iG1,iG2,iG3]+
                                    Q2[i,q,iG1,iG2,iG3]*R_xxx[I,p,r,s,iG1,iG2,iG3])

                            integrand += A8[p,q,r,s,t,u,v,w]*c1*c2
                        end
                        F3[idx] += wG_detJ[iG1,iG2,iG3]*integrand
                    end
                end
            end

        end
    end
    
    return F3
end


function calc_H3_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},δ::Array{Float64,2},A8::Array{Float64,8},
        nonzero_A8::Array{Int64,1},base8_list::Array{Array{Int64,1},1},
        R_x::Array{Float64,5},R_xx::Array{Float64,6},R_xxx::Array{Float64,7},Q2::Array{Float64,5},
        Q3::Array{Float64,6},Q4::Array{Float64,7})
    H3 = zeros(Float64,dim*Nall,dim*Nall)
    
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
                                        for α in nonzero_A8
                                            p,q,r,s,t,u,v,w = base8_list[α]

                                            c1 = (R_x[I,p,iG1,iG2,iG3]*R_xxx[M,q,r,s,iG1,iG2,iG3]+
                                                2*R_xx[I,p,r,iG1,iG2,iG3]*R_xx[M,q,s,iG1,iG2,iG3]+
                                                R_xxx[I,p,r,s,iG1,iG2,iG3]*R_x[M,q,iG1,iG2,iG3])

                                            c2_1 = 0.0
                                            c2_2 = 0.0
                                            for j in 1:dim
                                                c2_1 += (Q4[j,t,v,w,iG1,iG2,iG3]*
                                                            Q2[j,u,iG1,iG2,iG3])
                                                c2_2 += (Q3[j,u,w,iG1,iG2,iG3]*
                                                            Q3[j,t,v,iG1,iG2,iG3])
                                            end
                                            c2 = c2_1+c2_2

                                            c3 = (R_x[I,p,iG1,iG2,iG3]*Q4[i,q,r,s,iG1,iG2,iG3]+
                                                2.0*R_xx[I,p,r,iG1,iG2,iG3]*Q3[i,q,s,iG1,iG2,iG3]+
                                                R_xxx[I,p,r,s,iG1,iG2,iG3]*Q2[i,q,iG1,iG2,iG3])

                                            c4 = (R_xxx[M,t,v,w,iG1,iG2,iG3]*Q2[m,u,iG1,iG2,iG3]+
                                                2.0*R_xx[M,t,v,iG1,iG2,iG3]*Q3[m,u,w,iG1,iG2,iG3]+
                                                R_x[M,u,iG1,iG2,iG3]*Q4[m,t,v,w,iG1,iG2,iG3])

                                            integrand += A8[p,q,r,s,t,u,v,w]*(δ[i,m]*c1*c2+c3*c4)
                                        end
                                        H3[idx1,idx2] += wG_detJ[iG1,iG2,iG3]*integrand
                                    end
                                end
                            end

                        end
                    end

                    for i in 1:dim
                        for m in 1:dim
                            idx1 = (I-1)*dim+i
                            idx2 = (M-1)*dim+m
                            H3[idx2,idx1] = H3[idx1,idx2]
                        end
                    end
                end
            end

        end
    end
    return H3
end


function calc_F4_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},δ::Array{Float64,2},B::Array{Float64,6},
        nonzero_B::Array{Int64,1},base6_list::Array{Array{Int64,1},1},
        R_x::Array{Float64,5},R_xx::Array{Float64,6},R_xxx::Array{Float64,7},Q2::Array{Float64,5},
        Q3::Array{Float64,6},Q4::Array{Float64,7})
    F4 = zeros(Float64,dim*Nall)
    
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
                        for α in nonzero_B
                            p,q,r,s,t,u = base6_list[α]

                            c1 = R_x[I,p,iG1,iG2,iG3]*Q2[i,q,iG1,iG2,iG3]

                            c2_1 = 0.0
                            c2_2 = 0.0
                            for j in 1:dim
                                c2_1 += Q4[j,r,t,u,iG1,iG2,iG3]*Q2[j,s,iG1,iG2,iG3]
                                c2_2 += Q3[j,r,t,iG1,iG2,iG3]*Q3[j,s,u,iG1,iG2,iG3]
                            end
                            c2 = c2_1+c2_2

                            c3_1 = 0.0
                            for j in 1:dim
                                c3_1 += Q2[j,p,iG1,iG2,iG3]*Q2[j,q,iG1,iG2,iG3]
                            end
                            c3 = c3_1-δ[p,q]

                            c4 = (R_xxx[I,r,t,u,iG1,iG2,iG3]*Q2[i,s,iG1,iG2,iG3]+
                                    2*R_xx[I,r,t,iG1,iG2,iG3]*Q3[i,s,u,iG1,iG2,iG3]+
                                    R_x[I,s,iG1,iG2,iG3]*Q4[i,r,t,u,iG1,iG2,iG3])

                            integrand += B[p,q,r,s,t,u]*(1/2*c1*c2 + 1/4*c3*c4)
                        end
                        F4[idx] += wG_detJ[iG1,iG2,iG3]*integrand
                    end
                end
            end

        end
    end
    return F4
end


function calc_H4_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG_detJ::Array{Float64,3},δ::Array{Float64,2},B::Array{Float64,6},
        nonzero_B::Array{Int64,1},base6_list::Array{Array{Int64,1},1},
        R_x::Array{Float64,5},R_xx::Array{Float64,6},R_xxx::Array{Float64,7},Q2::Array{Float64,5},
        Q3::Array{Float64,6},Q4::Array{Float64,7})
    H4 = zeros(Float64,dim*Nall,dim*Nall)
    
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
                                        for α in nonzero_B
                                            p,q,r,s,t,u = base6_list[α]

                                            c1 = R_x[I,p,iG1,iG2,iG3]*R_x[M,q,iG1,iG2,iG3]

                                            c2_1 = 0.0
                                            c2_2 = 0.0
                                            for j in 1:dim
                                                c2_1 += (Q4[j,r,t,u,iG1,iG2,iG3]*
                                                            Q2[j,s,iG1,iG2,iG3])
                                                c2_2 += (Q3[j,r,t,iG1,iG2,iG3]*
                                                            Q3[j,s,u,iG1,iG2,iG3])
                                            end
                                            c2 = c2_1+c2_2

                                            c3_1 = 0.0
                                            for j in 1:dim
                                                c3_1 += (Q2[j,p,iG1,iG2,iG3]*Q2[j,q,iG1,iG2,iG3])
                                            end
                                            c3 = c3_1-δ[p,q]

                                            c4 = (R_xxx[I,r,t,u,iG1,iG2,iG3]*R_x[M,s,iG1,iG2,iG3]+
                                                2.0*R_xx[I,r,t,iG1,iG2,iG3]*R_xx[M,s,u,iG1,iG2,iG3]+
                                                R_x[I,s,iG1,iG2,iG3]*R_xxx[M,r,t,u,iG1,iG2,iG3])

                                            c5 = R_x[I,p,iG1,iG2,iG3]*Q2[i,q,iG1,iG2,iG3]

                                            c6 = (R_xxx[M,r,t,u,iG1,iG2,iG3]*Q2[m,s,iG1,iG2,iG3]+
                                                2.0*R_xx[M,r,t,iG1,iG2,iG3]*Q3[m,s,u,iG1,iG2,iG3]+
                                                R_x[M,s,iG1,iG2,iG3]*Q4[m,r,t,u,iG1,iG2,iG3])

                                            c7 = R_x[M,p,iG1,iG2,iG3]*Q2[m,q,iG1,iG2,iG3]

                                            c8 = (R_xxx[I,r,t,u,iG1,iG2,iG3]*Q2[i,s,iG1,iG2,iG3]+
                                                2.0*R_xx[I,r,t,iG1,iG2,iG3]*Q3[i,s,u,iG1,iG2,iG3]+
                                                    R_x[I,s,iG1,iG2,iG3]*Q4[i,r,t,u,iG1,iG2,iG3])

                                            integrand += (B[p,q,r,s,t,u]*(1/2*δ[i,m]*c1*c2 + 
                                                        1/4*δ[i,m]*c3*c4 + 1/2*c5*c6 + 1/2*c7*c8))
                                        end
                                        H4[idx1,idx2] += wG_detJ[iG1,iG2,iG3]*integrand
                                    end
                                end

                            end
                        end

                        for i in 1:dim
                            for m in 1:dim
                                idx1 = (I-1)*dim+i
                                idx2 = (M-1)*dim+m
                                H4[idx2,idx1] = H4[idx1,idx2]
                            end
                        end
                    end
                end
            end

        end
    end
    return H4
end


function calc_F5_3d(dim::Int64,P::Array{Int64,1},N::Array{Int64,1},Nall::Int64,nip::Array{Int64,1},
        G::Array{Int64,1},wG::Array{Array{Float64,1},1},b0::Float64,detJsrf::Array{Float64,2},
        R_x::Array{Float64,5},R_xx::Array{Float64,6},Q2::Array{Float64,5},Q3::Array{Float64,6},
        normal::Array{Float64,3})
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
                            c1 += R_xx[I,p,q,G[1],iG2,iG3]*Q2[i,p,G[1],iG2,iG3]
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
        detJsrf::Array{Float64,2},R_x::Array{Float64,5},R_xx::Array{Float64,6},normal::Array{Float64,3})
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
                                            c1 += R_xx[I,p,q,G[1],iG2,iG3]*R_x[M,p,G[1],iG2,iG3]
                                            c2 += R_x[I,p,G[1],iG2,iG3]*R_xx[M,p,q,G[1],iG2,iG3]
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

end