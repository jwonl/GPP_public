import numpy as np
from .source import DistributedSource
from .detector import *
from .scenario import *
from .path import *
from .utilities import *
from scipy.linalg import cholesky
from scipy.special import log_ndtr
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
# for fast kronecker computations
import pykronecker as kn
import time
import sys
from scipy.special import xlogy
from tqdm.notebook import tqdm

class Imager:
    def __init__(self, scenario, use_svd=False, rank=None):
        self.scenario = scenario
        self.meas = scenario.counts
        self.cov = None  # Covariance for the GPP reconstruction
        self.recon_img = None
        self.use_svd = use_svd
        self.sys_mat_rank = rank
        self.sys_mat = SYS_MAT(scenario.sys_mat, use_svd=use_svd, rank=self.sys_mat_rank)

    def MLEM_recon(self, itr=None, use_bkg=False, draw_img=False):
        """Perform MLEM/regularized reconstruction
        Args:
            itr (int): Number of MLEM iterations. Defaults to None.
            kld_reg (bool, optional): When True, perform KLD MLEM. Defaults to False.
            lam (float, optional): Regularization parameter for KLD MLEM. Defaults to None.
            draw_img (bool, optional): Draw the reconstructed source image. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert itr is not None
        # Initialize the image vector x
        x = np.ones(shape=(self.scenario.n_vox, 1))
        if use_bkg is True:
            b = 1
            sens_b = self.meas.size * self.scenario.int_time
            comp_numerator_b = (self.scenario.int_time * self.meas).T
        else:
            b = 0
        # Sensitivity
        sens = self.sys_mat.sum(axis=0)
        sens = sens.reshape(-1, 1)
        # This is the numerator of the comparerator
        comp_numerator = (self.sys_mat * self.meas).T
        
        time_elapsed = 0

        for i in range(itr):
            itr_start = time.time()
            # comparerator
            comp_denom = self.sys_mat @ x + b * self.scenario.int_time
            # comp_denom = self.sys_mat @ x
            comp = (comp_numerator / (comp_denom).T).sum(axis=1, keepdims=True)
            x = x / sens * comp
            if use_bkg is True:
                comp_b = np.sum(comp_numerator_b / (comp_denom).T)
                b = b / sens_b * comp_b
            itr_end = time.time()
            time_elapsed += itr_end - itr_start
            sys.stdout.write("\rMELM Iterations : (%d/%d)" % (i, itr))
            sys.stdout.flush()
        self.fp = self.sys_mat @ x + b * self.scenario.int_time
        self.recon_img = DistributedSource(src_str=x.sum() / 3.7e10)
        self.recon_img.src_dist = np.reshape(x, self.scenario.dims) / 3.7e10
        if draw_img is True:
            draw_2D_img(self.scenario, srcs=[self.recon_img])
        return {"img": np.reshape(x, self.scenario.dims)}

    def gpp_recon(
        self,
        lam=None,
        draw_img=False,
        cov=None,
        chol_cov=None,
        cov_kernel="squared_exponential",
        sigma=None,
        alpha=None,
        mixture_weights=None,
        mixture_scales=None,
        mixture_means=None,
        n_mixtures = None,
        KL_expansion=False,
        KL_rank=None,
        kronecker=False,
        start_from_prev=False,
        block_idx = None,
        block_sigma=None,
        block_lam=None,
        eps=1e-11,
        ftol=1e-9,
        iprint=5,
        verbose=False
    ):
        cov_t_start = time.time()
        if cov is None:
            if kronecker is True:
                #Use kronecker product of the covariance matrix in each imaging dimension to speed up the computation
                # FIXME this part of the code needs to be fixed to handle multiple different kernel functions and kernel construction methods more elegantly
                r_sqrd_list  = [
                    squareform(pdist(self.scenario.y_ctr.reshape(-1,1), metric="sqeuclidean")),
                    squareform(pdist(self.scenario.x_ctr.reshape(-1,1), metric="sqeuclidean")),
                    squareform(pdist(self.scenario.z_ctr.reshape(-1,1), metric="sqeuclidean"))
                    ]
                if cov_kernel == "squared_exponential":
                    cov = kn.KroneckerProduct(
                        [np.exp(-r_sqrd / (2 * sigma**2)) for r_sqrd in r_sqrd_list]
                        )
                    self.cov = cov
                elif cov_kernel == "exponential":
                    cov = kn.KroneckerProduct(
                        [np.exp(-np.sqrt(r_sqrd) / (sigma)) for r_sqrd in r_sqrd_list]
                        )
                    self.cov = cov
                elif cov_kernel == "rational_quadratic":
                    assert alpha is not None
                    cov = kn.KroneckerProduct(
                    [(1 + (r_sqrd / (2 * alpha * sigma**2))) ** (-alpha) for r_sqrd in r_sqrd_list]
                    )
                    self.cov = cov
                elif cov_kernel == "spectral_mixture":
                    # mixture weights, means, scales shaped n x d 
                    # where n is the number of mixture components and d is the dimension (x,y,z)
                    assert mixture_weights is not None
                    assert mixture_means is not None
                    assert mixture_scales is not None
                    assert mixture_weights.shape[0] == n_mixtures
                    assert mixture_scales.shape[0] == n_mixtures
                    assert mixture_means.shape[0] == n_mixtures
                    # construct SM covariance matrix in each dimension
                    cov = kn.KroneckerProduct([((mixture_weights[:,i]**2).reshape(1,1,-1) * np.exp(-2 * np.pi**2 * r_sqrd[:,:,None]/((mixture_scales[:,i].reshape(1,1,-1)**2))) * np.cos(2 * np.pi * np.sqrt(r_sqrd[:,:,None]) * mixture_means[:,i].reshape(1,1,-1))).sum(-1) for i, r_sqrd in enumerate(r_sqrd_list)])
                    self.cov = cov
                else:
                    print(cov_kernel)
                    raise ValueError("cov_kernel not specified")
                print(cov)
                cov_diag = self.cov.diag().reshape(-1,1)
                del(r_sqrd_list)
            else:
                r_sqrd = pdist(self.scenario.vox_ctr.reshape(-1, 3), metric="sqeuclidean")
                r_sqrd = squareform(r_sqrd)
                if cov_kernel == "squared_exponential":
                    cov = np.exp(-r_sqrd / (2 * sigma**2))
                elif cov_kernel == "rational_quadratic":
                    assert alpha is not None
                    cov = (1 + (r_sqrd / (2 * alpha * sigma**2))) ** (-alpha)
                del r_sqrd
                if block_idx is not None:
                    r_sqrd_block = pdist(self.scenario.vox_ctr.reshape(-1, 3)[block_idx.ravel()], metric="sqeuclidean")
                    r_sqrd_block = squareform(r_sqrd_block)
                    cov_block = np.exp(-r_sqrd_block / (2 * block_sigma**2))
                    block_cov = np.zeros_like(cov)
                    
                    withinblock_idx = np.outer(block_idx.ravel(), block_idx.ravel())
                    outofbloxk_idx = np.outer(np.logical_not(block_idx.ravel()), np.logical_not(block_idx.ravel()))
                    block_cov[withinblock_idx] = cov_block.ravel()
                    block_cov[outofbloxk_idx] = cov[outofbloxk_idx]
                    cov = block_cov
                    lam=np.ones(shape=(cov.shape[0], 1)) * lam
                    lam[block_idx.ravel()] = block_lam
                self.cov = cov
                cov_diag = np.diag(self.cov).reshape(-1,1)
        else:
            self.cov = cov
            cov_diag = np.diag(self.cov).reshape(-1,1)
        cov_t_finish = time.time()
        if verbose:
            print(f"Covariance construction time {cov_t_finish-cov_t_start:.2f}s")
        # For numerical stability, add eps diagonally and bump up the eigenvalues
        if chol_cov is None:
            cholesky_t_start = time.time()
            if kronecker is True:
                chol_cov = kn.KroneckerProduct([cholesky(cov_dim + eps * np.eye(cov_dim.shape[0]), lower=True) for cov_dim in self.cov.As])
            else:
                chol_cov = cholesky(self.cov + eps * np.eye(self.cov.shape[0]), lower=True)
            cholesky_t_finish = time.time()
            #print(f"Covariance matrix size {self.cov.shape}")
            if verbose:
                print(f"Cholesky decomposition time {cholesky_t_finish-cholesky_t_start:.2f}s")
            if start_from_prev is True:
                xi_w_init = self.xi_w.ravel() 
            else:
                xi_w_init = np.random.normal(size=(self.scenario.n_vox))
        opt_t_start = time.time()
        result = minimize(
            fun=gpp_f,
            x0=xi_w_init,
            args=(self.meas, sigma, lam, self.cov, chol_cov, cov_diag, self.sys_mat, kronecker),
            method="L-BFGS-B",
            # bounds = Bounds(0, np.inf, keep_feasible=False),
            jac=True,
            options={
                "iprint": iprint,
                "ftol": ftol,
                },
        )
        opt_t_finish = time.time()
        if verbose:
            print(f"Optimization time is {opt_t_finish-opt_t_start:.2f}")
        xi = chol_cov @ result.x.reshape(-1, 1)
        xi = xi if kronecker is False else xi.reshape(-1, 1)
        reconed_img = inv_link(
            lam, xi, self.cov, cov_diag=cov_diag
        ).reshape(-1, 1)
        self.fp = self.sys_mat @ reconed_img
        self.recon_img = DistributedSource(src_str=reconed_img.sum() / 3.7e10)
        self.recon_img.src_dist = np.reshape(reconed_img, self.scenario.dims) / 3.7e10
        self.xi_w = result.x.reshape(-1, 1)
        if draw_img is True:
            draw_2D_img(self.scenario, srcs=[self.recon_img])
        # TODO the following return will be removed
        return {
            "img": np.reshape(reconed_img, self.scenario.dims),
            "xi": np.reshape(xi, self.scenario.dims),
            "xi_w": self.xi_w,
            "cov": self.cov,
            "chol_cov": chol_cov,
            "cov_size": self.cov.shape,
            "lam" : lam,
            "cov_time":cov_t_finish-cov_t_start,
            "opt_time": opt_t_finish - opt_t_start,
            "results": result,
            "tot_time":opt_t_finish-cov_t_start,
        }
    
    def draw_reconed_img(self, **kwargs):
        assert self.recon_img is not None
        return draw_2D_img(self.scenario, srcs=[self.recon_img], **kwargs)

    def draw_forward_proj(self, saveimg=False, fname=None):
        count_data = [
            {"counts": self.fp_counts.sum(axis=1), "label": "True forward projection"},
            {"counts": self.counts.sum(axis=1), "label": "Measured counts"}
        ]
        plot_count_data(
            count_data=count_data, int_time=self.int_time, saveimg=saveimg, fname=fname
        )
        pass

    def draw_reconed_fp(self, saveimg=False, fname=None, **kwargs):
        count_data = [
            {
                "counts": self.scenario.counts.sum(axis=1),
                "kwargs": {"label": "Measured counts", **kwargs},
            },
            {
                "counts": self.fp,
                "kwargs": {"label": "Forward projection", **kwargs},
            },
        ]

        plot_count_data(
            count_data=count_data,
            int_time=self.scenario.int_time,
            saveimg=saveimg,
            fname=fname,
        )
        pass

    def pcn_mcmc(self, beta, n_samples, sigma, lam, cov_kernel=None, alpha=None, kronecker=False, starting_point=None, eps=1e-10, block_idx=None,
            block_sigma=None, block_lam=None,):
        """
        Perform preconditioned Crank-Nicolson Markov Chain Monte Carlo based on the gpp_recon parameters.
        For more information, refer to the Wikipedia page https://en.wikipedia.org/wiki/Preconditioned_Crank–Nicolson_algorithm
        and MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster, http://arxiv.org/abs/1202.0709.
        """
        # Initialize a matrix to hold the chain of samples.
        samples = np.zeros((self.scenario.n_vox, n_samples))
        # First, we perform gpp reconstruction to find the starting point.
        _ = self.gpp_recon(sigma=sigma, lam=lam, draw_img=True, cov_kernel=cov_kernel, alpha=alpha, kronecker=kronecker, block_idx=block_idx, block_sigma=block_sigma, block_lam=block_lam,)
        
        cov = _["cov"]
        chol_cov = _["chol_cov"]
        cov_diag = _["cov"].diag() if kronecker is True else np.diag(_["cov"])
        cov_diag = cov_diag.reshape(-1, 1)
        if block_idx is not None:
            lam_combined = np.ones((cov.shape[0],1)) * lam
            lam_combined[block_idx.ravel()] = block_lam
            lam=lam_combined
        if starting_point is None:
            # When the starting point is not provided, we start from the modal point (MAP estimate)
            x0 = _["xi"].reshape(-1, 1)
            samples[:, 0] = x0.ravel()
        else:
            assert starting_point.size == self.scenario.n_vox
            samples[:, 0] = starting_point.ravel()

        n_accept = 0
        mcmc_time_start = time.perf_counter()
        for i in tqdm(range(1, n_samples)):
            # Propose a new sample
            v = np.sqrt(1 - beta**2) * samples[:, i - 1].reshape(
                -1, 1
            ) + beta * (chol_cov @ (
                np.random.normal(size=(self.scenario.n_vox,1)))
            ).reshape(-1,1)
            # Calculate the acceptance probability
            accept_prob = pcn_acceptance_prob(
                samples[:, i - 1].reshape(-1, 1),
                v,
                lam,
                cov_diag,
                self.meas,
                self.sys_mat,
            )
            if np.random.rand() < accept_prob:
                samples[:, i] = v.ravel()
                n_accept += 1
            else:
                samples[:, i] = samples[:, i - 1]
            accept_prob = n_accept / i
            sys.stdout.write("\rAcceptance probability : %f" % accept_prob)
            sys.stdout.flush()
        return samples

    def type2_optimization(self, cov_kernel, param_init, n_mixtures=None, cov=None, method = "L-BFGS-B", eps=1e-7, kronecker=False, start_from_prev=False, block_idx=None, draw_img=False):
        print("draw_img in type2_optimization", draw_img)
        compute_lml = True
        if (cov_kernel == "squared_exponential") | (cov_kernel == "exponential"):
            param_init[0] = np.log(param_init[0])#sigma
            param_init[1] = np.log(param_init[1])#lambda
            if len(param_init) > 2:
                assert block_idx is not None
                param_init[2] = np.log(param_init[2])
                param_init[3] = np.log(param_init[3])
            bounds = None
        elif cov_kernel == "rational_quadratic":
            param_init[0] = np.log(param_init[0])#sigma
            param_init[1] = np.log(param_init[1])#alpha
            param_init[2] = np.log(param_init[2])#lambda
            bounds = None
        result = minimize(
            fun=type2_f,
            x0=param_init,
            args=(self, cov_kernel, n_mixtures, cov, eps, compute_lml, kronecker, start_from_prev, block_idx, draw_img),
            method=method,
            bounds = bounds,
            jac='2-point',
            options={"iprint": 1,
                      "ftol": 1e-9,
                        "finite_diff_rel_step":1e-2
                        }
                        ,
        )
        return result
    
    def laplace_approx(
            self,
            sigma=None,
            lam=None,
            cov_kernel="squared_exponential",
            alpha=None,
            mixture_weights=None,
            mixture_scales=None,
            mixture_means=None,
            n_mixtures = None,
            cov=None,
            eps=1e-7,
            compute_lml=False,
            kronecker=False,
            start_from_prev=False,
            block_idx=None,
            block_sigma=None,
            block_lam=None,
            draw_img=False):
        """
        Perform Laplace approximation to normal approximate the true postetior.
        the log-Gaussian distribution at the modal point is approximated using
        second-order Taylor expansion at the modal point.
        """
        # First, find the modal point
        if (cov_kernel == "squared_exponential") | (cov_kernel == "exponential"):
            _ = self.gpp_recon(
                sigma=sigma, 
                lam=lam, 
                cov_kernel=cov_kernel,
                cov=cov,
                draw_img=draw_img,
                kronecker=kronecker,
                KL_expansion=False,
                KL_rank=None,
                start_from_prev=start_from_prev,
                block_idx=block_idx,
                block_sigma=block_sigma,
                block_lam=block_lam,
                eps=eps)
        elif cov_kernel == "rational_quadratic":
            _ = self.gpp_recon(
                sigma=sigma,
                alpha=alpha,
                lam=lam, 
                cov_kernel=cov_kernel,
                cov=cov,
                draw_img=draw_img,
                kronecker=kronecker,
                KL_expansion=False,
                KL_rank=None,
                start_from_prev=start_from_prev,
                block_idx=block_idx,
                block_sigma=block_sigma,
                block_lam=block_lam,
                eps=eps)
        elif cov_kernel == "spectral_mixture":
            print("mixture_weights", mixture_weights)
            print("mixture_scales", mixture_scales)
            print("mixture_means", mixture_means)
            _ = self.gpp_recon(
                mixture_weights=mixture_weights,
                mixture_scales=mixture_scales,
                mixture_means=mixture_means,
                n_mixtures=n_mixtures,
                lam=lam, 
                cov_kernel=cov_kernel,
                cov=cov,
                draw_img=draw_img,
                kronecker=kronecker,
                KL_expansion=False,
                KL_rank=None,
                start_from_prev=start_from_prev,
                block_idx=block_idx,
                block_sigma=block_sigma,
                block_lam=block_lam,
                eps=eps)
        # Then we compute the Hessian matrix at the modal point
        laplace_start = time.time()
        xi = np.reshape(_["xi"], (-1, 1))
        cov = _["cov"]
        chol_cov = _["chol_cov"]
        lam = _["lam"]
        x = np.reshape(_["img"], (-1, 1))
        cov_diag = np.reshape(np.diag(cov), (-1, 1)) if kronecker is False else cov.diag().reshape(-1,1)
        fp = self.sys_mat @ x
        dLdf = self.sys_mat.T @ (1-self.meas / fp)
        dfdxi = (
            1
            / (np.sqrt(2 * np.pi * cov_diag) * lam)
            * np.exp(lam * x - xi**2 / (2 * cov_diag))
        )
        if compute_lml is False:
            #FIXME fix it to accept svd of sys matrix
            d2Ldfdxi = ((self.meas / (fp**2)) * self.sys_mat).T @ (
                self.sys_mat * (dfdxi).reshape(1, -1)
            ) 
            # d2fdxi2 = 1/(2*lam)*((-xi/cov_diag)*norm.pdf(xi, scale=cov_diag)*norm.cdf(-1*np.abs(xi), scale=cov_diag)\
            #     -(norm.pdf(xi, scale=cov_diag)*norm.pdf(-1*np.abs(xi), scale=cov_diag)))/(norm.cdf(-1*np.abs(xi), scale=cov_diag))**2
            d2fdxi2 = dfdxi * (lam * dfdxi - xi / cov_diag)
            hessian = d2Ldfdxi * dfdxi + np.diag((d2fdxi2 * dLdf).ravel())
        else:
            hessian = None
            # d2Ldfdxi = ((self.meas / (fp**2)) * self.sys_mat).T @ (
            #     self.sys_mat * (dfdxi).reshape(1, -1)
            # ) 
            # # d2fdxi2 = 1/(2*lam)*((-xi/cov_diag)*norm.pdf(xi, scale=cov_diag)*norm.cdf(-1*np.abs(xi), scale=cov_diag)\
            # #     -(norm.pdf(xi, scale=cov_diag)*norm.pdf(-1*np.abs(xi), scale=cov_diag)))/(norm.cdf(-1*np.abs(xi), scale=cov_diag))**2
            # d2fdxi2 = dfdxi * (lam * dfdxi - xi / cov_diag)
            # #hessian = d2Ldfdxi * dfdxi + np.diag((d2fdxi2 * dLdf).ravel())
            # hessian = d2Ldfdxi * dfdxi
        #hessian = d2Ldfdxi * dfdxi
        hessian_2 = None
        #print("(d2fdxi2)", d2fdxi2)
        #print("logdet hessian", np.linalg.slogdet(hessian))
        print(f"Hessian computation time", time.time() - laplace_start)
        grad = None
        #grad = (dfdxi)
        # For numerical stability, we add a small number to the eigenvalues
        a = time.time()
        if compute_lml is False:
            approx_cov = np.linalg.inv(
                hessian
                + np.eye(hessian.shape[0]) * eps
                + np.linalg.inv(cov + np.eye(cov.shape[0]) * eps)
            ) if kronecker is False else np.linalg.inv(
                hessian
                + np.eye(hessian.shape[0]) * eps
                + np.linalg.inv(cov.to_array() + np.eye(cov.shape[0]) * eps))
        else:
            approx_cov=None
        
        print("Hessian inversion time", time.time() - a)
        if compute_lml is True:
            lml_start = time.time()
            if kronecker is True:
                # FIXME The whole calculation involving the Kronecker product
                # should be done in a more effiecient and stable manner
                # Compute the negative log marginal likelihood based on the laplace approximation
                # sign, logdet = np.linalg.slogdet(approx_cov)
                # cov_logdet = np.array([np.linalg.slogdet(cov_dim + eps * np.eye(cov_dim.shape[0]))[1] * (1/cov_dim.shape[0]) for cov_dim in cov.As])
                # cov_logdet *= np.array([cov_dim.shape[0] for cov_dim in cov.As]).prod()
                # cov_logdet = cov_logdet.sum()
                # neg_logpost = _['results'].fun # neg log posterior measured with xi_w
                
                #sign, logdet = np.linalg.slogdet(np.eye(self.cov.shape[0]) + (hessian @ self.cov)) # this line computes the lml in a conventional way
                if self.use_svd is True:
                    XT = chol_cov.T @ np.asfortranarray((self.sys_mat.svt * dfdxi.reshape(1,-1)).T)
                    Y = self.sys_mat.u.T * (np.sqrt(self.meas)/fp).reshape(1,-1)
                    sign, logdet = np.linalg.slogdet(np.eye(self.sys_mat.u.shape[1])+(XT.T @ XT) @ (Y@Y.T))
                else:
                    Whalf =  dfdxi.reshape(-1,1) * self.sys_mat.T * (np.sqrt(self.meas) / (fp)).reshape(1,-1)
                    WKhalf = Whalf.T @ _['chol_cov']
                    sign, logdet = np.linalg.slogdet(np.eye(Whalf.shape[1]) + (WKhalf@WKhalf.T)) # this line computes the lml using the Woodbury formula
                    #sign, logdet = np.linalg.slogdet(np.eye(cov.shape[1]) + (hessian@cov)) # this line computes the lml using the Woodbury formula
                    #sign, logdet = np.linalg.slogdet(np.eye(cov.shape[1]) + (np.diag(np.diag(Whalf@Whalf.T))+np.diag((dfdxi * (lam * dfdxi - xi / cov_diag) * dLdf).ravel()))@cov) # this line computes the lml using the Woodbury formula
                    #sign, logdet = np.linalg.slogdet(np.eye(cov.shape[1]) + (((Whalf@Whalf.T))+np.diag((dfdxi * (lam * dfdxi - xi / cov_diag) * dLdf).ravel()))@cov) # this line computes the lml using the Woodbury formula
                neg_logpost = _['results'].fun # neg log posterior measured with xi_w
                # Compute the negative log marginal likelihood according to arXiv:2202.11678v2
                nlml = 0.5 * logdet + neg_logpost 
                print(f"sigma {sigma}, lambda {lam}")
                print(f"neg_logpose {neg_logpost}")
                print(f"0.5 * logdet {0.5 * logdet}, sign{sign}")
                #print(f"cov_logdet {cov_logdet}, sign{sign}")
                print(f"nlml {nlml}")
            else:
                # Compute the negative log marginal likelihood based on the laplace approximation
                #sign, logdet = np.linalg.slogdet(approx_cov)
                #TODO implement the SVD case 
                Whalf =  dfdxi.reshape(-1,1) * self.sys_mat.T * (np.sqrt(self.meas) / (fp)).reshape(1,-1)
                WKhalf = Whalf.T @ _['chol_cov']
                sign, logdet = np.linalg.slogdet(np.eye(Whalf.shape[1]) + (WKhalf@WKhalf.T)) # this line computes the lml using the Woodbury formula
                #cov_sign, cov_logdet = np.linalg.slogdet(cov+np.eye(cov.shape[0]) * eps)
                neg_logpost = _['results'].fun # neg log posterior measured with xi_w
                # Compute the negative log marginal likelihood according to arXiv:2202.11678v2
                nlml = 0.5 * logdet + neg_logpost 
                print(f"sigma {sigma}, lambda {lam}, {block_sigma},{block_lam}")
                print(f"neg_logpose {neg_logpost}")
                print(f"0.5 * logdet {0.5 * logdet}, sign{sign}")
                #print(f"cov_logdet {cov_logdet}, sign{sign}")
                print(f"nlml {nlml}")
            print("lml compute time", time.time() - lml_start)
            return xi, approx_cov, (dfdxi, fp, hessian, hessian_2), nlml
        else:
            return xi, approx_cov, (hessian, grad, d2Ldfdxi, dfdxi, d2fdxi2, dLdf)
        
class SYS_MAT:
    def __init__(self, sys_mat, use_svd=False, u=None, svt=None, rank=None):
        self.sys_mat = sys_mat
        self.use_svd = use_svd

        if self.use_svd is True:
            if (u is None) and (svt is None):
                self.u, self.svt = self.compute_svd(rank)
            else:
                self.u = u
                self.svt = svt
        else:
            self.u = u
            self.svt = svt
    
    def compute_svd(self, rank):
        from sklearn.decomposition import randomized_svd
        u, s, vt = randomized_svd(self.sys_mat, rank)
        return u, s.reshape(-1,1) * vt
    
    def __matmul__(self, other):
        assert isinstance(other, np.ndarray)
        if self.use_svd is False:
            return self.sys_mat @ other
        else:
            assert self.u is not None
            return self.u @ (self.svt @ other)
        
    def __rmatmul__(self, other):
        assert isinstance(other, np.ndarray)
        if self.use_svd is False:
            return other @ self.sys_mat
        else:
            assert self.u is not None
            return (other @ self.u) @ self.svt
    @property
    def T(self):
        #transpose of the system matrix
        if self.u is not None and self.svt is not None:
            return SYS_MAT(self.sys_mat.T, self.use_svd, self.svt.T, self.u.T)
        else:
            return SYS_MAT(self.sys_mat.T, self.use_svd, None, None)
        
    def sum(self, axis=None, keepdims=False):
        if self.use_svd is False:
            return np.sum(self.sys_mat, axis=axis, keepdims=keepdims)
        
        else:
            if axis == 1:
                result  = self.u @ (self.svt @ np.ones((self.shape[1], 1)))
            
            elif axis == 0:
                result  = (np.ones((1, self.shape[0])) @ self.u) @ self.svt

            else:
                result = (self.u @ self.svt).sum(keepdims)
            if keepdims is False:
                return np.squeeze(result)
            return result
    def __mul__(self, other):
        if self.use_svd is False:
            return self.sys_mat * other
        else:
            if (other.shape[0] == self.shape[0]) and (other.shape[1] == 1):
                return SYS_MAT(self.sys_mat, self.use_svd, other * self.u, self.svt)
            elif (other.shape[1] == self.shape[1]) and (other.shape[0] == 1):
                return SYS_MAT(self.sys_mat, self.use_svd, self.u, self.svt * other)
            else:
                raise NotImplementedError
            
    def __rmul__(self, other):
        if self.use_svd is False:
            return other * self.sys_mat
        else:
            if (other.shape[0] == self.shape[0]) and (other.shape[1] == 1):
                return SYS_MAT(self.sys_mat, self.use_svd, other * self.u, self.svt)
            elif (other.shape[1] == self.shape[1]) and (other.shape[0] == 1):
                return SYS_MAT(self.sys_mat, self.use_svd, self.u, self.svt * other)
            else:
                raise NotImplementedError
    def __truediv__(self, other):
        if self.use_svd is False:
            return self.sys_mat / other
        else:
            if (other.shape[0] == self.shape[0]) and (other.shape[1] == 1):
                return SYS_MAT(self.sys_mat, self.use_svd, self.u / other, self.svt)
            elif (other.shape[1] == self.shape[1]) and (other.shape[0] == 1):
                return SYS_MAT(self.sys_mat, self.use_svd, self.u, self.svt / other)
            else:
                raise NotImplementedError
            
    def __rtruediv__(self, other):
        if self.use_svd is False:
            return other / self.sys_mat
        else:
            if (other.shape[0] == self.shape[0]) and (other.shape[1] == 1):
                return SYS_MAT(self.sys_mat, self.use_svd, other / self.u, self.svt)
            elif (other.shape[1] == self.shape[1]) and (other.shape[0] == 1):
                return SYS_MAT(self.sys_mat, self.use_svd, self.u, other / self.svt)
            else:
                raise NotImplementedError
    @property
    def shape(self):
        return self.sys_mat.shape
    __array_priority__ = 10000

def type2_f(params, imager, cov_kernel, n_mixtures, cov, eps, compute_lml, kronecker, start_from_prev, block_idx, draw_img, block_sigma=None, block_lam=None):
    # FIXME this should be fixed to accept different hyperparameters for different covariance functions
    if (cov_kernel == "squared_exponential") | (cov_kernel == "exponential"):
        sigma = np.exp(params[0])#sigma
        lam = np.exp(params[1])#lambda
        if block_idx is not None:
            block_sigma = np.exp(params[2])#sigma
            block_lam = np.exp(params[3])#lambda
        # sigma = params[0]#sigma
        # lam = params[-1]#lambda
        result = imager.laplace_approx(sigma=sigma, lam=lam, cov_kernel=cov_kernel, cov=cov, eps=eps, compute_lml=compute_lml, kronecker=kronecker, start_from_prev=start_from_prev, block_idx=block_idx, block_sigma=block_sigma, block_lam=block_lam, draw_img=draw_img)
        lml = result[3]
    elif cov_kernel == "rational_quadratic":
        sigma = np.exp(params[0])#sigma
        alpha = np.exp(params[1])#k
        lam = np.exp(params[-1])#lambda
        result = imager.laplace_approx(sigma=sigma, alpha=alpha, lam=lam, cov_kernel=cov_kernel, cov=cov, eps=eps, compute_lml=compute_lml, kronecker=kronecker, start_from_prev=start_from_prev, block_idx=block_idx, block_sigma=block_sigma, block_lam=block_lam, draw_img=draw_img)
        lml = result[3]
    else:
        raise ValueError
    print(np.exp(params))
    return lml

def pcn_acceptance_prob(u, v, lam, cov_diag, meas, sys_mat):
    fp_u = sys_mat @ inv_link(lam, u, cov_diag)
    fp_v = sys_mat @ inv_link(lam, v, cov_diag)
    I_u = np.sum(fp_u - meas * np.log(fp_u))
    I_v = np.sum(fp_v - meas * np.log(fp_v))
    return np.minimum(1, np.exp(I_u - I_v))


# Loss function for the gpp algorithm
def gpp_f(xi_w, meas, sigma, lam, cov, chol_cov, cov_diag, sys_mat, kronecker):
    # xi_w is an 1d array
    xi = chol_cov @ xi_w.reshape(-1, 1)
    xi = xi.reshape(-1,1) if kronecker is True else xi
    x = inv_link(lam, xi, cov, cov_diag=cov_diag)
    fp = sys_mat @ x
    # loss log-likelihood
    loss_ll = np.sum(fp - xlogy(meas, fp))
    loss_lp = 0.5 * np.sum(xi_w**2)
    # Compute the gradient
    # Old way - slower
    #dLdf = np.sum(sys_mat.T, axis=1, keepdims=True) - sys_mat.T @ (meas / fp)
    # New way - faster
    dLdf = sys_mat.T @ (1-(meas / fp))

   
    # below is the gradient of the inverse link function with the infinite support laplacian distribution
    # dfdxi = 1/(2*lam)*norm.pdf(xi, scale=cov_diag)/norm.cdf(-1*np.abs(xi), scale=cov_diag)
    # dfdxi = (1/(np.sqrt(np.pi*2*cov_diag)) * np.exp(-1*xi**2/(2*cov_diag)))/(lam*(1-np.sign(xi)*erf(xi/(np.sqrt(2)*cov_diag))))
    # below is the gradient of the inverse link function with the non-negative support laplacian distribution (exponential distribution)
    dfdxi = (
        1
        / (np.sqrt(2 * np.pi * cov_diag) * lam)
        * np.exp(lam * x - xi**2 / (2 * cov_diag))
    )
    dxidxiw = chol_cov
    grad = ((dLdf * dfdxi).T @ dxidxiw + xi_w.T).T
    #print(x.shape)
    # print(f"ll loss:{loss_ll:.2f}, lp loss{loss_lp:.2f}, total loss{loss_ll+loss_lp:.2f}")

    return (loss_ll + loss_lp, grad.ravel())



def inv_link(lam, xi, cov, cov_diag=None):
    cov_diag = cov_diag if cov_diag is not None else np.reshape(np.diag(cov), (-1, 1))
    # log_ndtr for numerical stability instead of the error funciton
    return -1 / lam * log_ndtr(-xi)
