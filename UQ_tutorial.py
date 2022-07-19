def nll(y, rv_y):
    """Negative log likelihood for Tensorflow probability.

    Args:
        y: true data.
        rv_y: learned (predicted) probability distribution.
    """
    return -rv_y.log_prob(y)


def build_and_train_model(config: dict, n_components: int = 5, verbose: bool = 0):
    tf.keras.utils.set_random_seed(42)

    default_config = {
        "lstm_units": 128,
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "dropout_rate": 0,
        "num_layers": 1,
        "epochs": 20,
    }
    default_config.update(config)

    (X_train, y_train), (X_valid, y_valid), _, _ = load_data_prepared(
        n_components=n_components
    )

    layers = []
    for _ in range(default_config["num_layers"]):
        lstm_layer = tf.keras.layers.LSTM(
            default_config["lstm_units"],
            activation=default_config["activation"],
            recurrent_activation=default_config["recurrent_activation"],
            return_sequences=True,
        )
        dropout_layer = tf.keras.layers.Dropout(default_config["dropout_rate"])
        layers.extend([lstm_layer, dropout_layer])

    model = tf.keras.Sequential(
        [tf.keras.Input(shape=X_train.shape[1:])]
        + layers
        + [
            tf.keras.layers.Dense(n_components * 2),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :n_components],
                    scale=1e-3 + tf.math.softplus(0.05 * t[..., n_components:]),
                )
            ),
        ]
    )

    if verbose:
        model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=default_config["learning_rate"])
    model.compile(optimizer, loss=nll, metrics=[])

    history = model.fit(
        X_train,
        y_train,
        epochs=default_config["epochs"],
        batch_size=default_config["batch_size"],
        validation_data=(X_valid, y_valid),
        verbose=verbose,
    ).history

    return model, history

"""
define a NAS problem, add HPS into it
"""
from deephyper.problem import HpProblem
problem = HpProblem()
problem.load_data(load_data)

# Hyperparameter can also be search during neural architecture search
problem.add_hyperparameter((2, 64), "batch_size", default_value=64)
problem.add_hyperparameter(1e-4, 1e-2, "log-uniform"), "learning_rate", default_value=1e-3
)
optimizer=problem.add_hyperparameter(
        ["adam", "sgd", "adagrad", "rmsprop"], "optimizer", default_value="adam"
    )

problem.add_starting_point(**default_config)

num_epochs=20,

def run(config):

problem.loss(nll)
problem.objective("-val_loss")

"execute a genetic algorithm for the search"
from deephyper.search.hps import CBO

cbo_search = CBO(
    problem,
    evaluator,
    log_dir="cbo-results",
    initial_points=[problem.default_hp_configuration],
    random_state=42,
)
results = cbo_search.search(max_evals=100)

"perform UQ"

from deephyper.ensemble import UQBaggingEnsembleRegressor

ensemble = UQBaggingEnsembleRegressor(
    model_dir="agebo-results/save/model",
    loss=nll,  # default is nll
    size=5,
    verbose=True,
    ray_address="auto",
    num_cpus=1,
    num_gpus=1 if is_gpu_available else None,
    selection="caruana",
)

# ensemble.fit(X_valid, y_valid)

# print(f"Selected {len(ensemble.members_files)} members are: ", ensemble.members_files)

def ensemble_predict(X, kappa=1.96):
    y_pred_dist, var_aleatoric, var_epistemic = ensemble.predict_var_decomposition(X)

    mu = y_pred_dist.loc.numpy()
    std_aleatoric = np.sqrt(var_aleatoric)
    std_epistemic = np.sqrt(var_epistemic)

    mu_ci_aleatoric = mu + kappa * std_aleatoric
    mu_ci_epistemic = mu + kappa * std_epistemic
    mu_full = inverse_transform(mu)
    mu_ci_aleatoric = inverse_transform(mu_ci_aleatoric)
    mu_ci_epistemic = inverse_transform(mu_ci_epistemic)
    uq_aleatoric = np.abs(mu_full - mu_ci_aleatoric)
    uq_epistemic = np.abs(mu_full - mu_ci_epistemic)

    return mu_full, uq_aleatoric, uq_epistemic
