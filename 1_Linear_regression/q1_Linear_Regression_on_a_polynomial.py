def bgd(w, X_train, y_train, learning_rate=1e-3, num_iters=1):
    """Return weights and errors after given iterations of batch GD.
    
    Args:
        w -- M x 1 -- float: Weights.
        X_train -- N -- float: Inputs for training.
        y_train -- N -- float: Labels for training.
        learning_rate -- 1 -- float: Learning rate.
        num_iters -- 1 -- int: # of iterations.
    
    Returns:
        w -- M -- float: Updated weights.
        errors -- num_iters -- float: Errors after each iteration.
    """
    from numpy import concatenate, ones, newaxis, sum
    
    # Convert X_train to X to use it in GD,
    # i.e. concatenate [1, ... , 1] to its first row.
    N = len(y_train) # # of data points
    X = concatenate((ones((1, N)), X_train[newaxis, :]), axis=0)
    errors = []
    
    for iter in range(num_iters):
        w = w - learning_rate * X @ (w.T @ X - y_train[newaxis, :]).T / N
        MSE = sum((w.T @ X - y_train[newaxis, :]) ** 2) / N
        errors.append(MSE)
    
    return w, errors

def sgd(w, X_train, y_train, learning_rate=1e-3, num_iters=1):
    """Return weights and errors after given iterations of stochastic GD.
    
    One iteration uses all the data points.
    
    Args:
        w -- M x 1 -- float: Weights.
        X_train -- N -- float: Inputs for training.
        y_train -- N -- float: Labels for training.
        learning_rate -- 1 -- float: Learning rate.
        num_iters -- 1 -- int: # of iterations.
    
    Returns:
        w -- M -- float: Updated weights.
        errors -- num_iters -- float: Errors after each iteration.
    """
    from numpy import concatenate, ones, newaxis, sum
    
    # Convert X_train to X to use it in GD,
    # i.e. concatenate [1, ... , 1] to its first row.
    N = len(y_train) # # of data points
    X = concatenate((ones((1, N)), X_train[newaxis, :]), axis=0)
    errors = []
    
    for iter in range(num_iters):
        for idx in range(N):
            Xcur = X[:, idx, newaxis] # Data to use now.
            w = w - learning_rate * (w.T @ Xcur - y_train[idx]) * Xcur
        MSE = sum((w.T @ X - y_train[newaxis, :]) ** 2) / N
        errors.append(MSE)
    
    return w, errors

def main_a():
    """ Find the coefficients of a first degree polynomial using BGD and SGD.
    """
    from numpy import load, ones
    from matplotlib.pyplot import subplots, savefig, show
    
    # Load training data.
    X_train = load("data/q1xTrain.npy")
    y_train = load("data/q1yTrain.npy")
    
    # Set parameters.
    M = 2 # Define M, the # of features.
    w = ones((M, 1)) # Initialize the weights.
    num_iters = 100
    
    # Run BGD and SGD.
    wB, errB = bgd(w, X_train, y_train, learning_rate=1.5, num_iters=num_iters)
    wS, errS = sgd(w, X_train, y_train, learning_rate=0.1, num_iters=num_iters)
    
    # Find convergence iteration of BGD and SGD.
    # Define convergence if error <= 0.2
    # or decrease of error <= 0.0001.
    for idx in range(len(errB) - 1):
        if errB[idx] > 0.2 and errB[idx + 1] <= 0.2:
            thB = idx + 1
            break
        elif errB[idx] - errB[idx + 1] < 0.0001:
            thB = idx + 1
            break
    for idx in range(len(errS) - 1):
        if errS[idx] > 0.2 and errS[idx + 1] <= 0.2:
            thS = idx + 1
            break
        elif errS[idx] - errS[idx + 1] < 0.0001:
            thS = idx + 1
            break
    
    # Print weights and plot errors of BGD and SGD.
    print("Problem 1-(a)")
    print("Weight of BGD after {} iterations: {}".format(num_iters, wB.T))
    print("Weight of SGD after {} iterations: {}".format(num_iters, wS.T))
    fig, ax = subplots()
    ax.plot(errB, label="BGD Error")
    ax.title.set_text("MSE of Batch and Stochastic Gradient Descent\nBGD Weights: {}\nSGD Weights: {}".format(wB.T, wS.T))
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean Squared Error")
    ax.scatter(thB, errB[thB], label="BGD Convergence: After {}th iteration".format(thB))
    ax.plot(errS, label="SGD Error")
    ax.scatter(thS, errS[thS], label="SGD Convergence: After {}th iteration".format(thS))
    ax.legend()
    savefig("q1-a-ii.png")
    show()

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

def CFS(Phi, y_train):
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

def main_b():
    """ Find the coefficients of an M degree polynomial
    and check if overfitting happens.
    """
    from numpy import load, sqrt
    from matplotlib.pyplot import subplots, savefig, show
    
    # Load training and test data.
    X_train = load("data/q1xTrain.npy")
    y_train = load("data/q1yTrain.npy")
    X_test = load("data/q1xTest.npy")
    y_test = load("data/q1yTest.npy")
    
    RMSE_train = []
    RMSE_test = []
    for degree in range(10): # Iterate over polynomial degrees
        # Find closed form solutions with different # of features.
        Phi_train = make_design_matrix(X_train, M=degree+1)
        w = CFS(Phi_train, y_train)
        
        # Get "sum of squared error" of training data.
        SSE_train = (w.T @ Phi_train.T - y_train.T) @ (Phi_train @ w - y_train) / 2
        
        # Get RMSE of training data.
        N_train = len(y_train)
        RMSE_train.append(sqrt(2 * SSE_train / N_train))
        
        # Get "sum of squared error" and RMSE of testing data.
        Phi_test = make_design_matrix(X_test, M=degree+1)
        SSE_test = (w.T @ Phi_test.T - y_test.T) @ (Phi_test @ w - y_test) / 2
        N_test = len(y_test)
        RMSE_test.append(sqrt(2 * SSE_test / N_test))
    
    _, best_idx = min((val, idx) for (idx, val) in enumerate(RMSE_test))
    
    # Plot RMSE of training and testing data.
    print("Problem 1-(b)")
    fig, ax = subplots()
    ax.title.set_text("RMSE Plot for Under- and Over-fitting Check")
    ax.set_xlabel("M")
    ax.set_ylabel("RMSE")
    ax.plot(RMSE_train, label="Training Data")
    ax.plot(RMSE_test, label="Testing Data")
    ax.scatter(best_idx, RMSE_test[best_idx], label="{}th degree is the best".format(best_idx), color="orange")
    ax.legend()
    print("{}th degree is the best".format(best_idx))
    savefig("q1-b-i.png")
    show()

def CFS_ridge(Phi, reg, y_train):
    """ Return weights of ridge regression using closed form solution.
    
    Args:
        Phi -- N x M -- float: Design matrix.
        reg -- 1 -- float: Regularization coefficient.
        y_train -- N -- float: Labels for training.
    
    Returns:
        w -- M -- float: Updated weights.
    """
    from numpy import identity
    from numpy.linalg import pinv
    
    # Calculate closed form solution.
    w = pinv(Phi.T @ Phi + reg * identity(len(Phi.T))) @ (Phi.T @ y_train)
    
    return w

def main_c():
    """ Apply regularization.
    """
    from numpy import load, arange, sqrt
    from matplotlib.pyplot import subplots, savefig, show
    
    # Load training and test data.
    X_train = load("data/q1xTrain.npy")
    y_train = load("data/q1yTrain.npy")
    X_test = load("data/q1xTest.npy")
    y_test = load("data/q1yTest.npy")
    
    RMSE_train = []
    RMSE_test = []
    reg_list = [0] + [10 ** exp for exp in range(-8, 1)]
    for reg in reg_list:
        # Find closed form solutions of ridge regression.
        Phi_train = make_design_matrix(X_train, M=10)
        w = CFS_ridge(Phi_train, reg, y_train)
        
        # Get "sum of squared error" of training data.
        SSE_train = (w.T @ Phi_train.T - y_train.T) @ (Phi_train @ w - y_train) / 2
        
        # Get RMSE of training data.
        N_train = len(y_train)
        RMSE_train.append(sqrt(2 * SSE_train / N_train))
        
        # Get "sum of squared error" and RMSE of testing data.
        Phi_test = make_design_matrix(X_test, M=10)
        SSE_test = (w.T @ Phi_test.T - y_test.T) @ (Phi_test @ w - y_test) / 2
        N_test = len(y_test)
        RMSE_test.append(sqrt(2 * SSE_test / N_test))
    
    _, best_idx = min((val, idx) for (idx, val) in enumerate(RMSE_test))
    
    # Plot RMSE of training and testing data.
    reg_list = [str(elem) for elem in reg_list]
    print("Problem 1-(c)")
    fig, ax = subplots()
    ax.title.set_text("RMSE Plot to Check Regularization Effect")
    ax.set_xlabel("Regularization Coefficient")
    ax.set_ylabel("RMSE")
    ax.plot(reg_list, RMSE_train, label="Training Data")
    ax.plot(reg_list, RMSE_test, label="Testing Data")
    ax.scatter(best_idx, RMSE_test[best_idx], label="Reg. coeff. {} is the best".format(reg_list[best_idx]), color="orange")
    ax.legend()
    print("Regularization coefficient {} is the best".format(reg_list[best_idx]))
    savefig("q1-c-i.png")
    show()

if __name__ == "__main__":
    main_a()
    main_b()
    main_c()