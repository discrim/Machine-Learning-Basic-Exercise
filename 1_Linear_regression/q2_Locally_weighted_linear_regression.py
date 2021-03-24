def make_design_matrix(X, M):
    """ Return a design matrix.
    
    Args:
        X -- N -- float: Inputs for training or testing.
        M -- 1 -- int: # of features to use.
    
    Returns:
        Phi -- N x M -- float: Design matrix.
    """
    assert M > 0 and type(M) == int, "M should be a positive integer."
    
    from numpy import zeros
    
    N = len(X)
    Phi = zeros((N, M))
    for col in range(M):
        Phi[:, col] = X ** col
    
    return Phi

def CFS_LR(Phi, y_train):
    """ Return weights of linear regression using closed form solution.
    
    Args:
        Phi -- N x M -- float: Design matrix.
        y_train -- N -- float: Labels for training.
    
    Returns:
        w -- M -- float: Updated weights.
    """
    from numpy.linalg import pinv
    
    # Calculate closed form solution.
    w = pinv(Phi.T @ Phi) @ (Phi.T @ y_train)
    
    return w

def make_R(query, x_train, tau):
    """ Return R for locally weighted linear regression.
    
    Args:
        query -- 1 -- float: Input that I want to focus on.
        x_train -- N -- float:
            Entire train input to determine the local weight of
            each train input with respect to the current query.
        tau -- 1 -- float: Bandwidth parameter.
    
    Returns:
        R -- -- float: Local weights for weighted linear regression.
    """
    from numpy import zeros, exp, diag
    R = zeros((len(x_train), len(x_train)))
    
    # Calculate each element of R using give formula and vectorize them.
    R = [exp(- (query - x_train[idx]) ** 2 / (2 * tau ** 2)) for idx in range(len(x_train))]
    
    # Make a diagonal matrix using the vector.
    R = diag(R)
    
    return R

def CFS_LW(Phi, R, y_train):
    """ Return weights of locally weighted linear regression 
        using closed form solution.
    
    Args:
        Phi -- N x M -- float: Design matrix.
        R: -- N x N -- float: Local weights for a given query.
        y_train -- N -- float: Labels for training.
    
    Returns:
        w -- M -- float: Updated weights.
    """
    from numpy.linalg import pinv
    
    # Calculated the closed form solution for locally weighted LR.
    w = pinv(Phi.T @ R @ Phi) @ (Phi.T @ (R @ y_train))
    
    return w

def main():
    """ Run linear regression with various local weight parameters.
    """
    from numpy import load, linspace
    from matplotlib.pyplot import subplots, savefig, show
    
    # load data.
    x_train = load("data/q2x.npy")
    y_train = load("data/q2y.npy")
    
    # Run unweighted linear regression.
    # Find closed form solution for unweighted linear regression.
    Phi = make_design_matrix(x_train, 2)
    w = CFS_LR(Phi, y_train)
    y_result = w @ Phi.T
    
    # Plot the data and the line, and save as file.
    fig, ax = subplots()
    ax.title.set_text("Linear Regression without Local Weighting\ny = {:.3f} + {:.3f}x".format(w[0], w[1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter(x_train, y_train, label="Data", color="black")
    ax.plot(x_train, y_result, label="Unweighted LR Line")
    ax.legend()
    savefig("q2-d-i.png")
    
    # Run locally weighted linear regression with tau = 0.8.
    # Prepare queries.
    xmin, xmax = min(x_train), max(x_train)
    xran = xmax - xmin
    queries = linspace(xmin, xmax, num=50)
    
    # Iterate over each query.
    y_result = []
    for query in queries:
        # Find closed for solution for locally weighted linear regression.
        R = make_R(query, x_train, 0.8)
        w = CFS_LW(Phi, R, y_train)
        y_result.append(w[0] + w[1] * query)
    
    # Plot the curve and save as file.
    ax.title.set_text("Local Weighting Linear Regression")
    ax.plot(queries, y_result, label="Locally Weighted LR Curve, tau=0.8")
    ax.legend()
    savefig("q2-d-ii.png")
    
    # Run locally weighted linear regression using various taus.
    # Nest another iteration over four tau = 0.1, 0.3, 2, 10.
    taus = [0.1, 0.3, 2, 10]
    for tau in taus:
        y_result = []
        for query in queries:
            # Find closed for solution for locally weighted linear regression.
            R = make_R(query, x_train, tau)
            w = CFS_LW(Phi, R, y_train)
            y_result.append(w[0] + w[1] * query)
        
        # Plot the curve and save as file.
        ax.plot(queries, y_result, label="Locally Weighted LR Curve, tau={}".format(tau))
    ax.title.set_text("Local Weighted Linear Regression")
    ax.legend()
    savefig("q2-d-iii.png")
    show()

if __name__ == "__main__":
    main()