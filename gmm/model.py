import numpy as np
import matplotlib.pyplot as plt


from data.data import X


def normal(X, mu, sigma):
    """
    X: (N x d), data points
    pi: (C), mixture component weights 
    mu: (C x d), mixture component means
    sigma: (C x d x d), mixture component covariance matrices
    """

    X = np.expand_dims(X, axis=1)  # N, 1, d
    Xmu = X-mu  # N, C, d

    A = np.expand_dims(sigma, axis=0)  # 1, C, d, d
    b = Xmu  # N, C, d
    sigma_inv_Xmu = np.linalg.solve(A, b)  # N, C, d

    exponent = -0.5 * np.einsum('ncd,ncd->nc', Xmu, sigma_inv_Xmu)  # N, C
    denominator = np.linalg.det(2*np.pi*sigma)  # C
    return np.exp(exponent) / np.sqrt(denominator)


def E_step(X, pi, mu, sigma):
    """
    Performs E-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    pi: (C), mixture component weights 
    mu: (C x d), mixture component means
    sigma: (C x d x d), mixture component covariance matrices
    
    Returns:
    gamma: (N x C), probabilities of clusters for objects
    """
    p_x_given_t = normal(X, mu, sigma)
    p_x_t = p_x_given_t * pi  # N, c

    p_x = np.sum(p_x_t, axis=-1)  # N
    gamma = p_x_t / np.expand_dims(p_x, axis=-1)

    return gamma


def M_step(X, gamma):
    """
    Performs M-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)  
    
    Returns:
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    """
    N = X.shape[0]  # number of objects

    pi = np.sum(gamma, axis=0) / N  # C

    mu = np.matmul(np.transpose(gamma), X)  # C, d
    mu = mu / (N*pi[:, np.newaxis])

    X = np.expand_dims(X, axis=1)  # N, 1, d
    Xmu = X-mu  # N, C, d
    sigma = np.einsum('ncd,nc,ncD->cdD', Xmu, gamma, Xmu)  # C, d, d
    sigma = sigma / (N*pi[:, np.newaxis, np.newaxis])

    return pi, mu, sigma


def compute_vlb(X, pi, mu, sigma, gamma):
    """
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)  
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    
    Returns value of variational lower bound
    """

    log_pi = np.expand_dims(np.log(pi), axis=0)  # 1, C

    norm = normal(X, mu, sigma)  # N, C

    logs = (log_pi + np.log(norm) - np.log(gamma))  # N, C

    loss = np.sum(gamma*logs)
    return loss


def train_EM(X, C, tol=1e-1, max_iter=100, restarts=10):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *tol* reached
    or *max_iter* iterations were made.
    
    X: (N, d), data points
    C: int, number of clusters
    '''
    N = X.shape[0]  # number of objects
    d = X.shape[1]  # dimension of each object
    best_loss = np.NINF
    best_pi = best_mu = best_sigma = None
    sigma0 = np.repeat(np.expand_dims(np.eye(d), axis=0), C, axis=0)

    for _ in range(restarts):
        print("_________________")
        pi = np.ones(C) / C
        mu = X[np.random.choice(N, C, False), :]
        sigma = sigma0
        prev_loss = np.NINF
        try:
            for step in range(max_iter):
                gamma = E_step(X, pi, mu, sigma)
                pi, mu, sigma = M_step(X, gamma)
                loss = compute_vlb(X, pi, mu, sigma, gamma)
                if loss > best_loss:
                    best_loss = loss
                    best_pi = pi
                    best_mu = mu
                    best_sigma = sigma
                if loss < prev_loss:  # Must increase
                    print("ERRRROOOOOORRRRRRRRR")
                    break
                if loss-prev_loss < tol:  # Absolute convergence
                    print(f"Converged after {step}: {loss}")
                    break
                prev_loss = loss
        except np.linalg.LinAlgError:
            print("Singular matrix: components collapsed")

    return best_loss, best_pi, best_mu, best_sigma


def main():
    # gamma = E_step(X, pi0, mu0, sigma0)
    # print(gamma)

    best_loss, best_pi, best_mu, best_sigma = train_EM(X, 2)

    gamma = E_step(X, best_pi, best_mu, best_sigma)
    labels = gamma.argmax(1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
