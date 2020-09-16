# single qubit VQE

## importing necessary libraries 


```python
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute
from qiskit.aqua.components.optimizers import COBYLA
import matplotlib.pyplot as plt

backend = Aer.get_backend("qasm_simulator")
NUM_SHOTS = 10000

```

## initiating the target random state


```python
np.random.seed(5321) #initiating a random vector to be our target vector  
random_distribution = np.random.rand(2) #initiating the random vector with 2 rows
random_distribution /= sum(random_distribution) #normalizing the vector

```

## creating a circuit with the parameters needing to be optimized (parameterized circuit)/variational form U(theta)


```python
def para_circuit(params):
    cr =ClassicalRegister(1,name="c") 
    qr=QuantumRegister(1,name="q")
    qc=QuantumCircuit(cr,qr)
    qc.u3(params[0],params[1],params[2],qr[0])
    qc.measure(qr,cr[0])
    return qc
```

## Creating a function for extracting a probability distribution from a circuit/ state


```python
def probability_distribution(counts):
    output_dist=[v/NUM_SHOTS for v in counts.values()]
    if len(output_dist) == 1:
        output_dist.append(0)
    return output_dist

```

## defining objective function 


```python
def objective_fn(params):
    qc=para_circuit(params)
    results=execute(qc,backend,shots=NUM_SHOTS).result()
    our_model_dist= probability_distribution(results.get_counts(qc))
    cost_function=sum([np.abs(our_model_dist[i]-random_distribution[i]) for i in range(2)])
                      
    return cost_function
                    
    
```

# Using an optimizer to find the parameters value for the model distribution to match the random distribution

## the optimizer works as this: 
## first we define a number of iterations and an error tolerance value
## we then ask the optimizer to optimize the "parameters" of a certain "circuit" with respect to the "objective function" but the parameters are encoded in the objective function along with the circuit .. the objective function does the simulation of the circuit for the optimizer so the optimizer now only needs the objective functions and an initial point  


```python
optimizer =COBYLA(maxiter=200, tol=0.0005)
initial_param=np.random.rand(3)
Param_result=optimizer.optimize(num_vars=3, objective_function=objective_fn, initial_point=initial_param)


```

## after the optimizer has been trained we compare the distribution of our model to the random distribution


```python
qc=para_circuit(Param_result[0])
result=execute(qc,backend,shots=NUM_SHOTS).result()
counts=result.get_counts()
model_distribution=probability_distribution(counts)
print("Random Distribution:", random_distribution)
print("Model Distribution:", model_distribution)
print("Output Error (Manhattan Distance):", Param_result[1])
print("Parameters Found:", Param_result[0])

```

    Random Distribution: [0.17434917 0.82565083]
    Model Distribution: [0.1789, 0.8211]
    Output Error (Manhattan Distance): 0.010301664307097724
    Parameters Found: [4.00103129 2.45309069 0.4431488 ]
    


