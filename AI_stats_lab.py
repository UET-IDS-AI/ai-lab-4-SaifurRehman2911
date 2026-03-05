import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class GDResult:
    theta: np.ndarray
    losses: np.ndarray
    theta_path: np.ndarray


# =========================
# Q1 Gradient Descent
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:

    n, d = X.shape

    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()

    losses = []
    theta_path = []

    for _ in range(epochs):

        y_pred = X @ theta
        error = y_pred - y

        loss = np.mean(error ** 2)
        losses.append(loss)

        grad = (2 / n) * (X.T @ error)

        theta = theta - lr * grad

        theta_path.append(theta.copy())

    return GDResult(
        theta=theta,
        losses=np.array(losses),
        theta_path=np.array(theta_path)
    )


# =========================
# Visualization dataset
# =========================

def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:

    np.random.seed(seed)

    n = 100
    X_feature = np.random.randn(n)

    true_theta0 = 2
    true_theta1 = 3

    noise = np.random.randn(n) * 0.5

    y = true_theta0 + true_theta1 * X_feature + noise

    X = np.column_stack([np.ones(n), X_feature])

    result = gradient_descent_linreg(X, y, lr, epochs)

    return {
        "theta_path": result.theta_path,
        "losses": result.losses,
        "X": X,
        "y": y
    }


# =========================
# Q2 Diabetes with GD
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    data = load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])

    result = gradient_descent_linreg(X_train, y_train, lr, epochs)

    theta = result.theta

    train_pred = X_train @ theta
    test_pred = X_test @ theta

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q3 Analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    data = load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])

    d = X_train.shape[1]

    I = np.eye(d)

    theta = np.linalg.inv(
        X_train.T @ X_train + ridge_lambda * I
    ) @ X_train.T @ y_train

    train_pred = X_train @ theta
    test_pred = X_test @ theta

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q4 Comparison
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:

    gd_train_mse, gd_test_mse, gd_train_r2, gd_test_r2, gd_theta = diabetes_linear_gd(
        lr, epochs, test_size, seed
    )

    an_train_mse, an_test_mse, an_train_r2, an_test_r2, an_theta = diabetes_linear_analytical(
        1e-8, test_size, seed
    )

    theta_l2_diff = np.linalg.norm(gd_theta - an_theta)

    cosine_sim = np.dot(gd_theta, an_theta) / (
        np.linalg.norm(gd_theta) * np.linalg.norm(an_theta)
    )

    return {
        "theta_l2_diff": theta_l2_diff,
        "train_mse_diff": abs(gd_train_mse - an_train_mse),
        "test_mse_diff": abs(gd_test_mse - an_test_mse),
        "train_r2_diff": abs(gd_train_r2 - an_train_r2),
        "test_r2_diff": abs(gd_test_r2 - an_test_r2),
        "theta_cosine_sim": cosine_sim,
    }
