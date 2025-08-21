# Bayesian Neural Networks (BNNs) Examples

The demo of the code supplementing [The Architecture and Evaluation of Bayesian Neural Networks](https://arxiv.org/abs/2503.11808) paper for which we implement and train Bayesian Neural Networks using JAX, NumPyro, and Flax.


The `BNNs_examples.ipynb` notebook demonstrates how to implement and train Bayesian Neural Networks using two different inference methods:
- **Mean-field Variational Inference (VI)**
- **Hamiltonian Monte Carlo (HMC)**


### BNN Class
The main `BNN` class provides a interface for:
- Model initialization with customizable architectures (ReLU vs. Sigmoid, depth, width)
- Prior specification for network parameters
- Training with both VI (mean-field ADVI) and MCMC (HMC with No U-Turn Sampler) methods
- Obtaining posterior and predictive distributions


### Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Open the Jupyter notebook: `jupyter notebook BNNs_examples.ipynb`
3. Experiment!

### Usage Example

```python
# Initialize BNN with specific architecture
bnn = BNN(hidden_dims=[200], activation='ReLU')

# Train with Variational Inference
bnn.doVI(x_train, y_train, n_iterations=10000, lr=5e-3)

# Generate posterior predictions
bnn.posterior_predictive_vi_dist(x_test, n_samples=1000)

# Train with MCMC (alternative)
bnn.doMCMC(x_train, y_train, n_warmup=1000, n_samples=1000, n_chains=1)
bnn.predict_mcmc(x_test)
```