from torch import matmul, cat,tril,triu, diagonal

def one_component_conditioning(c,x=None,m=None,var = None, reduce=True,cov2cor=False):
    # var if it is already extracted from C
    d = c.size(-1)
    v = diagonal(c,dim1=-2,dim2=-1) if var is None else var
    c_v = c/v.unsqueeze(-1)
    if x is None:
        m_c = 0 if m is None else m.unsqueeze(-2)-m.unsqueeze(-1)*c_v
    else:
        m_c = x.unsqueeze(-1)*c_v if m is None else m.unsqueeze(-2)+((x-m).unsqueeze(-1))*c_v
    c_3d = c_v.unsqueeze(-1)*c.unsqueeze(-2)
    c_c = c.unsqueeze(-3)-c_3d
    if cov2cor:
        stds = diagonal(c_c,dim1=-2,dim2=-1).sqrt()
        m_c = m_c/stds
        c_c = (c_c/stds.unsqueeze(-1))/stds.unsqueeze(-2)
    if reduce:
        m_cond = tril(m_c,-1)[...,:,:-1]+triu(m_c,1)[...,:,1:]
        l = tuple(remove_slice(c_c[...,i,:,:],i,-1) for i in range(d))
        c_cond = cat(tuple(remove_slice(l[i],i,-2).unsqueeze(-3) for i in range(d)),-3)
    else:
        m_cond = m_c
        c_cond = c_c
    return m_cond, c_cond

def remove_slice(M,i,dim=-1):
    if dim==-1:
        return cat((M[...,:i],M[...,i+1:]),-1)
    elif dim==-2:
        return cat((M[...,:i,:],M[...,i+1:,:]),-2)
    if dim==-3:
        return cat((M[...,:i,:,:],M[...,i+1:,:,:]),-3)
    elif dim==0:
        return cat((M[:i,...],M[i+1:,...]),0)
    elif dim==1:
        return cat((M[:,:i,...],M[:,i+1:,...]),1)
    elif dim==2:
        return cat((M[:,:,:i,...],M[:,:,i+1:,...]),2)
    else:
        raise NotImplementedError("dim must be -1, -2, -3, 0, 1 or 2.")

def two_components_conditioning(c,x=None,m=None,var = None,cov2cor=False):
    d = c.size(-1)
    m_cond1, c_cond1 = one_component_conditioning(c,x,m,var=var,reduce=True,cov2cor=False) # cov2cor is at the end
    m_c2 = []
    c_c2 = []
    for i in range(1,d):
        m_ci = m_cond1[...,i,:]
        C_ci = c_cond1[...,i,:,:]
        x_mi  = 0 if x is None else remove_slice(x,i,-1)
        for j in range(i):
            x_mimj = 0 if x is None else  remove_slice(x_mi,j,-1)
            x_mij =  0 if x is None else  x_mi[...,j]
            m_cimj = remove_slice(m_ci,j,-1)
            m_cij = m_ci[...,j]
            C_cijj = C_ci[...,j,j]
            C_cimj = remove_slice(C_ci,j,-2)
            C_cimjmj = remove_slice(C_cimj,j,-1)
            C_cimjj = C_cimj[...,:,j]
            m_cicj = m_cimj + ((x_mij-m_cij)/C_cijj).unsqueeze(-1)*C_cimjj
            C_cicj = C_cimjmj - matmul(C_cimjj.unsqueeze(-1),C_cimjj.unsqueeze(-1).transpose(-1,-2))/C_cijj.unsqueeze(-1).unsqueeze(-1)
            m_c2 += [m_cicj]
            c_c2 += [C_cicj]
    dd = d*(d-1)//2
    m_cond2 = cat(tuple(m_c2[i].unsqueeze(-2) for i in range(dd)),-2)
    c_cond2 = cat(tuple(c_c2[i].unsqueeze(-3) for i in range(dd)),-3)

    if cov2cor:
        stds1 = diagonal(c_cond1,dim1=-2,dim2=-1).sqrt()
        m_cond1 = m_cond1/stds1
        c_cond1 = (c_cond1/stds1.unsqueeze(-1))/stds1.unsqueeze(-2)
        stds2 = diagonal(c_cond2,dim1=-2,dim2=-1).sqrt()
        m_cond2 = m_cond2/stds2
        c_cond2 = (c_cond2/stds2.unsqueeze(-1))/stds2.unsqueeze(-2)
    return m_cond1, c_cond1, m_cond2, c_cond2
    

if __name__ == "__main__":
    """
    import torch
    d = 5
    bd = [2,3]
    values = torch.randn(*bd,d,requires_grad = True)
    A = torch.randn(*bd,d,d)
    C = torch.matmul(A,A.transpose(-2,-1))
    mean = torch.randn(*bd,d)

    mm = torch.empty(*bd,0,d-1)
    cc = torch.empty(*bd,0,d-1,d-1)
    for i in range(d):
        indices = torch.cat((torch.tensor(range(0,i),dtype=torch.int64),torch.tensor(range(i+1,d),dtype=torch.int64)),-1)
        
        m_mi = torch.index_select(mean, -1, indices)
        m_i = mean[...,i]
        x_mi = torch.index_select(values, -1, indices)
        x_i = values[...,i]
        S_ii = C[...,i,i]
        S_mimi = torch.index_select(torch.index_select(C, -1, indices), -2, indices)
        S_mii = torch.index_select(C, -2, indices)[...,:,i]
        m_cond = m_mi + ((x_i-m_i)/S_ii).unsqueeze(-1)*S_mii
        c_cond = S_mimi - torch.matmul(S_mii.unsqueeze(-1),S_mii.unsqueeze(-1).transpose(-1,-2))/S_ii.unsqueeze(-1).unsqueeze(-1)
        mm = torch.cat((mm,m_cond.unsqueeze(-2)),-2)
        cc = torch.cat((cc,c_cond.unsqueeze(-3)),-3)


    mm2, cc2 =  one_component_conditioning(C,values,mean,reduce=True)



    mm = torch.zeros(*bd,d*(d-1)//2,d-2)
    cc = torch.zeros(*bd,d*(d-1)//2,d-2,d-2)
    ij = 0
    for i in range(1,d):
        for j in range(i):
            n_indices = torch.cat((torch.tensor(range(0,j),dtype=torch.int64),torch.tensor(range(j+1,i),dtype=torch.int64),torch.tensor(range(i+1,d),dtype=torch.int64)),-1)
            indices = torch.tensor((i,j),dtype=torch.int64)
            m_mi = torch.index_select(mean, -1, n_indices)
            m_i = torch.index_select(mean, -1, indices)
            x_mi = torch.index_select(values, -1, n_indices)
            x_i = torch.index_select(values, -1, indices)
            S_ii = torch.index_select(torch.index_select(C, -1, indices), -2, indices)
            S_mimi = torch.index_select(torch.index_select(C, -1, n_indices), -2, n_indices)
            S_mii = torch.index_select(torch.index_select(C, -1, indices), -2, n_indices)
            m_cond = m_mi.unsqueeze(-1) + torch.matmul(torch.matmul(S_mii,torch.inverse(S_ii)),(x_i-m_i).unsqueeze(-1))
            c_cond = S_mimi - torch.matmul(torch.matmul(S_mii,torch.inverse(S_ii)),S_mii.transpose(-2,-1))
            mm[...,ij,:] = m_cond.squeeze(-1)
            cc[...,ij,:,:] = c_cond

            ij = ij+1




    _,_,mm_D, cc_D =  two_components_conditioning(C,values,mean)


    """

    import torch
    d = 5
    bd = [2,3]
    values = torch.randn(*bd,d,requires_grad = True)
    A = torch.randn(*bd,d,d)
    C = torch.matmul(A,A.transpose(-2,-1))
    mean = torch.randn(*bd,d)
    mm2, cc2 =  one_component_conditioning(C,values,mean,reduce=False)
    from torch.autograd import grad
    print(mm2.mean())
    print(grad(mm2.mean(),(values)))